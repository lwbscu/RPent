# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
import inspect

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

import robots.behavior.schemas as behavior_schemas
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    CLOSE_SPEC,
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    MOVE_TO_SPEC,
    NAVIGATE_TO_SPEC,
    OBSERVE_SPEC,
    OPEN_SPEC,
    PI0_NAV_PICK_SPEC,
    PIXEL_TO_WORLD_SPEC,
    PRESS_SPEC,
    PUBLIC_PRIMITIVE_ENTRYPOINTS,
    PUBLIC_TOOL_CONTRACTS,
    ROTATE_WRIST_SPEC,
    SAVE_ROBOT_STATE_CHECKPOINT_SPEC,
    behavior_tool_specs_for_task,
)
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives

EXPECTED_TOOL_NAMES = (
    "pi0_nav_pick",
    "observe",
    "pixel_to_world",
    "move_to",
    "rotate_wrist",
    "close",
    "open",
    "press",
    "save_robot_state_checkpoint",
    "navigate_to",
)

LEGACY_PUBLIC_NAMES = {
    "finish",
    "inspect_post_pick_state",
    "inspect_toggle_geometry",
    "post_pick_close_press_gripper",
    "post_pick_recenter_held_button",
    "post_pick_direct_finger_toggle",
    "post_success_hold_frames",
}


def test_default_tool_call_budget_is_350_with_explicit_override(tmp_path):
    signature = inspect.signature(BehaviorPrimitives)
    assert signature.parameters["max_tool_calls"].default == 350

    default_toolkit = BehaviorToolkit(primitives_kwargs={"output_dir": tmp_path})
    overridden_toolkit = BehaviorToolkit(
        primitives_kwargs={"output_dir": tmp_path, "max_tool_calls": 777}
    )

    assert default_toolkit._max_tool_calls == 350
    assert default_toolkit._primitives.max_tool_calls == 350
    assert overridden_toolkit._max_tool_calls == 777
    assert overridden_toolkit._primitives.max_tool_calls == 777


def _validator(spec):
    return Draft202012Validator(spec["input_schema"])


def _visual_hand_check(hand: str) -> dict[str, str]:
    return {
        "camera": "head",
        "frame_id": f"head:current:{hand}",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _depth_probe(
    *,
    frame_id: str = "head:7:fresh",
    u: int = 320,
    v: int = 240,
    depth_window_px: int = 7,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "u": u,
        "v": v,
        "depth_window_px": depth_window_px,
        "assessment": "target_point_visually_confirmed",
    }


def test_public_primitive_surface_v2_appends_navigation_to_immutable_v1():
    assert CURRENT_PUBLIC_TOOL_CONTRACT_VERSION == 2
    assert tuple(PUBLIC_TOOL_CONTRACTS) == (1, 2)
    assert PUBLIC_TOOL_CONTRACTS[1] == EXPECTED_TOOL_NAMES[:-1]
    assert PUBLIC_TOOL_CONTRACTS[2] == EXPECTED_TOOL_NAMES
    assert PUBLIC_TOOL_CONTRACTS[2][:-1] == PUBLIC_TOOL_CONTRACTS[1]
    assert BEHAVIOR_TOOL_NAMES == EXPECTED_TOOL_NAMES
    assert len(BEHAVIOR_TOOL_NAMES) == len(set(BEHAVIOR_TOOL_NAMES)) == 10
    assert LEGACY_PUBLIC_NAMES.isdisjoint(BEHAVIOR_TOOL_NAMES)
    assert tuple(PUBLIC_PRIMITIVE_ENTRYPOINTS) == EXPECTED_TOOL_NAMES
    assert set(behavior_schemas.__all__).isdisjoint(
        {
            "BEHAVIOR_FINISH_SPEC",
            "INSPECT_POST_PICK_STATE_SPEC",
            "INSPECT_TOGGLE_GEOMETRY_SPEC",
            "POST_PICK_CLOSE_PRESS_GRIPPER_SPEC",
            "POST_PICK_RECENTER_HELD_BUTTON_SPEC",
            "POST_PICK_DIRECT_FINGER_TOGGLE_SPEC",
            "POST_SUCCESS_HOLD_FRAMES_SPEC",
        }
    )


def test_public_entrypoints_use_neutral_primitive_methods():
    assert PUBLIC_PRIMITIVE_ENTRYPOINTS == {
        "pi0_nav_pick": "BehaviorPrimitives.pi0_nav_pick",
        "observe": "BehaviorPrimitives.observe",
        "pixel_to_world": "BehaviorPrimitives.pixel_to_world",
        "move_to": "BehaviorPrimitives.move_to",
        "rotate_wrist": "BehaviorPrimitives.rotate_wrist",
        "close": "BehaviorPrimitives.close",
        "open": "BehaviorPrimitives.open",
        "press": "BehaviorPrimitives.press",
        "save_robot_state_checkpoint": (
            "BehaviorPrimitives.save_robot_state_checkpoint"
        ),
        "navigate_to": "BehaviorPrimitives.navigate_to",
    }


def test_nav_pick_schema_requires_unbounded_exact_chunk_count():
    schema = PI0_NAV_PICK_SPEC["input_schema"]
    assert set(schema["properties"]) == {
        "instruction",
        "chunks",
    }
    assert schema["required"] == ["instruction", "chunks"]
    assert schema["additionalProperties"] is False
    chunks_schema = schema["properties"]["chunks"]
    assert chunks_schema["type"] == "integer"
    assert chunks_schema["minimum"] == 1
    assert "maximum" not in chunks_schema
    validator = _validator(PI0_NAV_PICK_SPEC)
    validator.validate({"instruction": "pick up the task object", "chunks": 1})
    validator.validate({"instruction": "pick up the task object", "chunks": 10**12})
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValidationError):
            validator.validate(
                {
                    "instruction": "pick up the task object",
                    "chunks": invalid,
                }
            )
    with pytest.raises(ValidationError):
        validator.validate(
            {
                "instruction": "pick up the task object",
                "chunks": 1,
                "max_chunks": 128,
            }
        )
    with pytest.raises(ValidationError):
        validator.validate({"instruction": "pick up the task object"})
    description = PI0_NAV_PICK_SPEC["description"].lower()
    for capability in ("navigation", "grasping", "pressing"):
        assert capability in description
    assert "no fixed maximum" in description
    assert "requested work bound" in description
    assert "first" not in description
    assert "exactly once" not in description
    assert "single permitted" not in description


def test_nav_pick_schema_is_unchanged_by_analytic_rgbd_hand_geometry():
    schema = PI0_NAV_PICK_SPEC["input_schema"]
    serialized = repr(schema)

    for analytic_only_field in (
        "depth_probe",
        "target_point_camera_xyz_m",
        "target_to_palm_m",
        "target_to_grip_point_m",
        "target_to_finger_roots_m",
        "hand_geometry",
        "hand",
        "role",
        "resolved_hand",
        "release_visual_check",
        "visual_hand_check",
    ):
        assert analytic_only_field not in serialized


def test_observe_schema_uses_only_public_physical_camera_names():
    schema = OBSERVE_SPEC["input_schema"]
    assert schema["required"] == ["camera"]
    assert set(schema["properties"]) == {"camera", "frame_review", "depth_probe"}
    assert schema["properties"]["camera"]["enum"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]
    frame_review = schema["properties"]["frame_review"]
    assert frame_review["type"] == "object"
    assert set(frame_review["properties"]) == {"frame_id", "assessment"}
    assert frame_review["required"] == ["frame_id", "assessment"]
    assert frame_review["additionalProperties"] is False
    assert frame_review["properties"]["frame_id"]["minLength"] == 1
    assert frame_review["properties"]["assessment"]["enum"] == [
        "target_bearing_surface_confirmed",
        "opposite_surface_confirmed",
        "side_or_indeterminate",
    ]
    depth_probe = schema["properties"]["depth_probe"]
    assert depth_probe["type"] == "object"
    assert set(depth_probe["properties"]) == {
        "frame_id",
        "u",
        "v",
        "depth_window_px",
        "assessment",
    }
    assert depth_probe["required"] == [
        "frame_id",
        "u",
        "v",
        "depth_window_px",
        "assessment",
    ]
    assert depth_probe["additionalProperties"] is False
    assert depth_probe["properties"]["frame_id"]["minLength"] == 1
    assert depth_probe["properties"]["depth_window_px"]["minimum"] == 1
    assert depth_probe["properties"]["depth_window_px"]["maximum"] == 31
    assert depth_probe["properties"]["assessment"]["const"] == (
        "target_point_visually_confirmed"
    )
    assert schema["allOf"] == [{"not": {"required": ["frame_review", "depth_probe"]}}]
    assert schema["additionalProperties"] is False


def test_observe_schema_accepts_capture_exact_frame_review_or_exact_depth_probe():
    validator = _validator(OBSERVE_SPEC)
    validator.validate({"camera": "head"})
    validator.validate(
        {
            "camera": "left_wrist",
            "frame_review": {
                "frame_id": "left_wrist:7:fresh",
                "assessment": "target_bearing_surface_confirmed",
            },
        }
    )
    for camera in ("head", "left_wrist", "right_wrist"):
        validator.validate({"camera": camera, "depth_probe": _depth_probe()})
    for invalid in (
        {"camera": "held_wrist"},
        {"camera": "press_wrist"},
        {
            "camera": "head",
            "frame_review": {"assessment": "side_or_indeterminate"},
        },
        {
            "camera": "head",
            "frame_review": {
                "frame_id": "head:7:fresh",
                "assessment": "unsupported",
            },
        },
        {
            "camera": "head",
            "frame_review": {
                "frame_id": "head:7:fresh",
                "assessment": "opposite_surface_confirmed",
                "extra": True,
            },
        },
        {
            "camera": "head",
            "depth_probe": {
                key: value for key, value in _depth_probe().items() if key != "frame_id"
            },
        },
        {
            "camera": "head",
            "depth_probe": {
                **_depth_probe(),
                "assessment": "target_point_probably_visible",
            },
        },
        {
            "camera": "head",
            "depth_probe": {**_depth_probe(), "extra": True},
        },
        {
            "camera": "head",
            "frame_review": {
                "frame_id": "head:7:fresh",
                "assessment": "side_or_indeterminate",
            },
            "depth_probe": _depth_probe(),
        },
    ):
        with pytest.raises(ValidationError):
            validator.validate(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_id", ""),
        ("u", 1.5),
        ("u", True),
        ("v", 1.5),
        ("v", False),
        ("depth_window_px", 0),
        ("depth_window_px", 32),
        ("depth_window_px", True),
    ],
)
def test_observe_depth_probe_schema_rejects_invalid_coordinates_or_window(field, value):
    with pytest.raises(ValidationError):
        _validator(OBSERVE_SPEC).validate(
            {
                "camera": "head",
                "depth_probe": {**_depth_probe(), field: value},
            }
        )


def test_observe_depth_probe_is_available_to_both_tasks_without_radio_leakage():
    radio_observe = behavior_tool_specs_for_task(TURNING_ON_RADIO_TASK_SPEC)["observe"][
        "input_schema"
    ]
    trash_observe = behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)["observe"][
        "input_schema"
    ]

    assert set(radio_observe["properties"]) == {
        "camera",
        "frame_review",
        "depth_probe",
    }
    assert set(trash_observe["properties"]) == {"camera", "depth_probe"}
    _validator(
        behavior_tool_specs_for_task(TURNING_ON_RADIO_TASK_SPEC)["observe"]
    ).validate({"camera": "head", "depth_probe": _depth_probe()})
    _validator(
        behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)["observe"]
    ).validate({"camera": "left_wrist", "depth_probe": _depth_probe()})
    with pytest.raises(ValidationError):
        _validator(
            behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)["observe"]
        ).validate(
            {
                "camera": "head",
                "frame_review": {
                    "frame_id": "head:7:fresh",
                    "assessment": "side_or_indeterminate",
                },
            }
        )


def test_pixel_to_world_schema_is_generic_and_frame_bound():
    schema = PIXEL_TO_WORLD_SPEC["input_schema"]
    assert set(schema["properties"]) == {
        "camera",
        "frame_id",
        "u",
        "v",
        "depth_window_px",
    }
    assert schema["required"] == ["camera", "frame_id", "u", "v"]
    assert schema["properties"]["camera"]["enum"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]
    assert schema["properties"]["depth_window_px"]["minimum"] == 1
    assert "target_kind" not in schema["properties"]
    assert "button_visibility" not in schema["properties"]
    description = PIXEL_TO_WORLD_SPEC["description"].lower()
    for field in ("world", "surface normal", "confidence", "projection_id"):
        assert field in description


def test_navigate_to_schema_supports_projection_or_explicit_relative_base_motion():
    schema = NAVIGATE_TO_SPEC["input_schema"]
    validator = _validator(NAVIGATE_TO_SPEC)
    assert schema["required"] == []
    assert len(schema["oneOf"]) == 2
    assert set(schema["properties"]) == {
        "projection_id",
        "navigation_visual_check",
        "relative_motion",
        "standoff_m",
    }
    assert schema["additionalProperties"] is False
    visual = schema["properties"]["navigation_visual_check"]
    assert visual["required"] == ["camera", "frame_id", "assessment"]
    assert visual["additionalProperties"] is False
    assert visual["properties"]["camera"]["const"] == "head"
    assert visual["properties"]["assessment"]["const"] == (
        "navigation_target_visually_confirmed"
    )
    assert schema["properties"]["standoff_m"] == {
        "type": "number",
        "default": 0.85,
        "minimum": 0.45,
        "maximum": 1.5,
    }
    assert "hand" not in repr(schema)
    assert "role" not in repr(schema)
    for forbidden in (
        "target_xyz",
        "delta_xyz",
        "frame",
        "chunks",
        "max_chunks",
        "visual_hand_check",
    ):
        assert forbidden not in schema["properties"]

    valid = {
        "projection_id": "projection-current",
        "navigation_visual_check": {
            "camera": "head",
            "frame_id": "head:7:fresh",
            "assessment": "navigation_target_visually_confirmed",
        },
    }
    validator.validate(valid)
    relative_valid = [
        {
            "relative_motion": {
                "kind": "translation",
                "direction": direction,
                "distance_m": 1.5,
            }
        }
        for direction in ("forward", "backward")
    ] + [
        {
            "relative_motion": {
                "kind": "rotation",
                "direction": direction,
                "angle_deg": 180.0,
            }
        }
        for direction in ("left", "right")
    ]
    for relative in relative_valid:
        validator.validate(relative)
    for invalid in (
        {key: value for key, value in valid.items() if key != "projection_id"},
        {
            **valid,
            "navigation_visual_check": {
                **valid["navigation_visual_check"],
                "camera": "left_wrist",
            },
        },
        {
            **valid,
            "navigation_visual_check": {
                **valid["navigation_visual_check"],
                "frame_id": "",
            },
        },
        {
            **valid,
            "navigation_visual_check": {
                **valid["navigation_visual_check"],
                "assessment": "target_point_visually_confirmed",
            },
        },
        {**valid, "hand": "left"},
        {**valid, "role": "held"},
        {**valid, "target_xyz": [0.0, 0.0, 0.0]},
        {**valid, "chunks": 1},
        {**valid, "max_travel_m": 0.0},
        {**valid, "standoff_m": 0.44},
        {**valid, "timeout_s": 0.0},
        {
            **valid,
            "relative_motion": {
                "kind": "translation",
                "direction": "forward",
                "distance_m": 0.2,
            },
        },
        {
            "relative_motion": {
                "kind": "translation",
                "direction": "forward",
                "distance_m": 0.0,
            }
        },
        {
            "relative_motion": {
                "kind": "translation",
                "direction": "left",
                "distance_m": 0.2,
            }
        },
        {
            "relative_motion": {
                "kind": "rotation",
                "direction": "right",
                "angle_deg": 181.0,
            }
        },
        {
            "relative_motion": {
                "kind": "rotation",
                "direction": "right",
                "angle_deg": 90.0,
                "distance_m": 0.2,
            }
        },
        {
            "relative_motion": {
                "kind": "translation",
                "direction": "backward",
                "distance_m": 0.2,
            },
            "max_travel_m": 1.0,
        },
    ):
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_move_to_schema_accepts_projection_or_relative_delta_only():
    schema = MOVE_TO_SPEC["input_schema"]
    validator = _validator(MOVE_TO_SPEC)
    assert "maximum" not in schema["properties"]["target"]["properties"]["standoff_m"]
    validator.validate(
        {
            "hand": "right",
            "target": {"projection_id": "projection-current", "standoff_m": 0.04},
            "visual_hand_check": _visual_hand_check("right"),
        }
    )
    validator.validate(
        {
            "hand": "left",
            "target": {
                "delta_xyz": [0.01, -0.02, 0.03],
                "frame": "eef",
            },
            "visual_hand_check": _visual_hand_check("left"),
        }
    )
    for invalid in (
        {
            "hand": "left",
            "target": {"projection_id": "p", "delta_xyz": [0, 0, 0]},
            "visual_hand_check": _visual_hand_check("left"),
        },
        {
            "hand": "left",
            "target": {"delta_xyz": [0, 0, 0]},
            "visual_hand_check": _visual_hand_check("left"),
        },
        {
            "hand": "left",
            "target": {"target_xyz": [0, 0, 0]},
            "visual_hand_check": _visual_hand_check("left"),
        },
        {
            "hand": "left",
            "target": {"target_quat_xyzw": [0, 0, 0, 1]},
            "visual_hand_check": _visual_hand_check("left"),
        },
        {
            "hand": "left",
            "target": {"projection_id": "p"},
            "visual_hand_check": _visual_hand_check("left"),
            "plan_only": True,
        },
        {
            "hand": "left",
            "target": {"projection_id": "p"},
            "visual_hand_check": _visual_hand_check("left"),
            "alignment_phase": "final",
        },
    ):
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_rotate_wrist_schema_requires_relative_axis_angle_and_visual_hand_check():
    schema = ROTATE_WRIST_SPEC["input_schema"]
    validator = _validator(ROTATE_WRIST_SPEC)
    assert set(schema["properties"]) == {
        "hand",
        "relative_axis_angle",
        "frame",
        "visual_hand_check",
    }
    validator.validate(
        {
            "hand": "right",
            "relative_axis_angle": [0, 0, 1, 0.2],
            "frame": "eef",
            "visual_hand_check": _visual_hand_check("right"),
        }
    )
    assert set(schema["required"]) == {
        "hand",
        "relative_axis_angle",
        "visual_hand_check",
    }
    assert "target_quat_xyzw" not in schema["properties"]
    with pytest.raises(ValidationError):
        validator.validate({"hand": "left"})
    with pytest.raises(ValidationError):
        validator.validate(
            {
                "hand": "left",
                "target_quat_xyzw": [0, 0, 0, 1],
            }
        )
    assert "plan_only" not in ROTATE_WRIST_SPEC["input_schema"]["properties"]


def test_all_analytic_specs_require_physical_hand_and_fresh_head_visual_check():
    requests = {
        "move_to": {
            "target": {"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
        },
        "rotate_wrist": {
            "relative_axis_angle": [0.0, 1.0, 0.0, 0.2],
        },
        "close": {},
        "open": {},
        "press": {
            "projection_id": "projection-current",
            "travel_m": 0.03,
        },
    }
    for spec in (MOVE_TO_SPEC, ROTATE_WRIST_SPEC, CLOSE_SPEC, OPEN_SPEC, PRESS_SPEC):
        schema = spec["input_schema"]
        assert schema["properties"]["hand"]["enum"] == ["left", "right"]
        assert "role" not in schema["properties"]
        assert "visual_hand_check" in schema["required"]
        visual_schema = schema["properties"]["visual_hand_check"]
        assert visual_schema["properties"]["camera"]["const"] == "head"
        assert visual_schema["properties"]["assessment"]["const"] == (
            "selected_hand_visually_confirmed"
        )
        validator = _validator(spec)
        base = requests[spec["name"]]
        for hand in ("left", "right"):
            validator.validate(
                {
                    "hand": hand,
                    **base,
                    "visual_hand_check": _visual_hand_check(hand),
                }
            )
        with pytest.raises(ValidationError):
            validator.validate({"hand": "left", **base})
        for legacy in ("held", "press"):
            with pytest.raises(ValidationError):
                validator.validate(
                    {
                        "hand": legacy,
                        **base,
                        "visual_hand_check": _visual_hand_check("left"),
                    }
                )
            with pytest.raises(ValidationError):
                validator.validate(
                    {
                        "role": legacy,
                        **base,
                        "visual_hand_check": _visual_hand_check("left"),
                    }
                )


def test_hand_schema_rejects_visual_hand_mismatch_for_every_analytic_tool():
    requests = {
        "move_to": {
            "target": {"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
        },
        "rotate_wrist": {
            "relative_axis_angle": [0.0, 1.0, 0.0, 0.2],
        },
        "close": {},
        "open": {},
        "press": {
            "projection_id": "projection-current",
            "travel_m": 0.03,
        },
    }
    for spec in (MOVE_TO_SPEC, ROTATE_WRIST_SPEC, CLOSE_SPEC, OPEN_SPEC, PRESS_SPEC):
        with pytest.raises(ValidationError):
            _validator(spec).validate(
                {
                    "hand": "left",
                    **requests[spec["name"]],
                    "visual_hand_check": _visual_hand_check("right"),
                }
            )


def test_visual_hand_check_schema_rejects_wrong_camera_assessment_or_shape():
    base = {
        "hand": "left",
        "relative_axis_angle": [0.0, 1.0, 0.0, 0.2],
    }
    for invalid in (
        {**_visual_hand_check("left"), "camera": "left_wrist"},
        {**_visual_hand_check("left"), "assessment": "hand_probably_visible"},
        {**_visual_hand_check("left"), "frame_id": ""},
        {**_visual_hand_check("left"), "extra": True},
    ):
        with pytest.raises(ValidationError):
            _validator(ROTATE_WRIST_SPEC).validate(
                {**base, "visual_hand_check": invalid}
            )


@pytest.mark.parametrize("spec", [CLOSE_SPEC, OPEN_SPEC])
def test_gripper_specs_use_physical_hand_visual_check(spec):
    schema = spec["input_schema"]
    assert schema["required"] == ["hand", "visual_hand_check"]
    _validator(spec).validate(
        {
            "hand": "left",
            "visual_hand_check": _visual_hand_check("left"),
        }
    )


def test_open_schema_accepts_only_exact_same_hand_release_visual_check():
    trash_open = behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)["open"]
    radio_open = behavior_tool_specs_for_task(TURNING_ON_RADIO_TASK_SPEC)["open"]
    validator = _validator(trash_open)
    release_check = {
        "camera": "head",
        "frame_id": "head:22:fresh",
        "selected_hand": "left",
        "assessment": "attached_object_fully_inside_receptacle_opening",
    }
    validator.validate(
        {
            "hand": "left",
            "visual_hand_check": _visual_hand_check("left"),
            "release_visual_check": release_check,
        }
    )
    for invalid in (
        {**release_check, "camera": "left_wrist"},
        {**release_check, "frame_id": ""},
        {**release_check, "selected_hand": "right"},
        {**release_check, "assessment": "object_probably_inside"},
        {**release_check, "extra": True},
    ):
        with pytest.raises(ValidationError):
            validator.validate(
                {
                    "hand": "left",
                    "visual_hand_check": _visual_hand_check("left"),
                    "release_visual_check": invalid,
                }
            )
    assert "release_visual_check" not in CLOSE_SPEC["input_schema"]["properties"]
    assert "release_visual_check" not in OPEN_SPEC["input_schema"]["properties"]
    assert "release_visual_check" not in radio_open["input_schema"]["properties"]
    assert "attached_object_fully_inside_receptacle_opening" not in repr(radio_open)


def test_press_schema_is_projection_bound_without_a_fixed_travel_ceiling():
    schema = PRESS_SPEC["input_schema"]
    assert set(schema["required"]) == {
        "hand",
        "projection_id",
        "travel_m",
        "visual_hand_check",
    }
    travel = schema["properties"]["travel_m"]
    assert travel["exclusiveMinimum"] == 0.0
    assert "maximum" not in travel
    _validator(PRESS_SPEC).validate(
        {
            "hand": "right",
            "projection_id": "projection-current",
            "travel_m": 0.3,
            "visual_hand_check": _visual_hand_check("right"),
        }
    )
    with pytest.raises(ValidationError):
        _validator(PRESS_SPEC).validate(
            {
                "hand": "right",
                "projection_id": "projection-current",
                "travel_m": 0.0,
                "visual_hand_check": _visual_hand_check("right"),
            }
        )


def test_checkpoint_schema_supports_only_visual_label_or_bound_terminal_failure():
    schema = SAVE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]
    assert set(schema["properties"]) == {"semantic_label", "terminal_failure"}
    assert schema["properties"]["semantic_label"]["maxLength"] == 128
    assert schema["required"] == []
    assert schema["additionalProperties"] is False
    declaration = {
        "condition": "radio_tipped_flat",
        "cause": "dropped_out_of_gripper",
        "camera": "head",
        "frame_id": "head:17:fresh",
    }
    _validator(SAVE_ROBOT_STATE_CHECKPOINT_SPEC).validate(
        {"terminal_failure": declaration}
    )
    for invalid in (
        {"terminal_failure": {**declaration, "condition": "attachment_lost"}},
        {"terminal_failure": {**declaration, "cause": "unknown"}},
        {"terminal_failure": {**declaration, "camera": "left"}},
        {"terminal_failure": {**declaration, "frame_id": ""}},
        {
            "terminal_failure": {
                **declaration,
                "unbound_claim": True,
            }
        },
    ):
        with pytest.raises(ValidationError):
            _validator(SAVE_ROBOT_STATE_CHECKPOINT_SPEC).validate(invalid)


def test_env_client_exposes_neutral_rpc_methods_without_legacy_public_methods():
    public_methods = {
        name
        for name, value in inspect.getmembers(BehaviorEnvClient, inspect.isfunction)
        if not name.startswith("_")
    }
    assert {
        "pi0_nav_pick_chunk_step",
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
    }.issubset(public_methods)
    assert LEGACY_PUBLIC_NAMES.isdisjoint(public_methods)
    assert "reset" in public_methods
    assert "reset" not in BEHAVIOR_TOOL_NAMES
