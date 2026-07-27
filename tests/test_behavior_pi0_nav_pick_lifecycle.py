from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.planner_executor import (
    official_task_success as planner_official_task_success,
)
from robots.behavior.schemas import PI0_NAV_PICK_SPEC, validate_action_chunk
from robots.behavior.tools import (
    BehaviorPrimitives,
)
from robots.behavior.tools import (
    official_task_success as tools_official_task_success,
)


def test_pi0_transport_still_requires_complete_action_chunks():
    chunk = np.zeros((32, 23), dtype=np.float32)
    assert validate_action_chunk(chunk, max_horizon=32).shape == (32, 23)
    for bad in (
        np.zeros((31, 23), dtype=np.float32),
        np.zeros((32, 22), dtype=np.float32),
    ):
        try:
            validated = validate_action_chunk(bad, max_horizon=32)
            if validated.shape != (32, 23):
                raise ValueError("not one complete VLA chunk")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid VLA chunk accepted: {bad.shape}")


def test_pi0_lifecycle_source_has_no_attempt_level_single_call_gate():
    source = inspect.getsource(BehaviorEnvFacade.prepare_vla_invocation)
    assert "_pi0_invocation_consumed" not in source
    assert "exactly once" not in source
    assert "single Pi0" not in source


def test_pi0_path_does_not_use_planner_single_arm_isolation():
    source = "\n".join(
        (
            inspect.getsource(BehaviorPrimitives.pi0_nav_pick),
            inspect.getsource(BehaviorEnvFacade.pi0_nav_pick_chunk_step),
        )
    )
    assert "capture_single_arm_isolation_reference" not in source
    assert "single_arm_isolation_report" not in source
    assert "isolation_reference" not in source


def test_pi0_public_contract_has_no_hand_selection_fields():
    properties = PI0_NAV_PICK_SPEC["input_schema"]["properties"]
    assert set(properties) == {
        "instruction",
        "chunks",
        "current_object_visual_check",
    }
    for forbidden in (
        "role",
        "hand",
        "visual_hand_check",
        "selected_hand",
        "projection_id",
        "navigation_visual_check",
        "standoff_m",
        "max_travel_m",
    ):
        assert forbidden not in properties

    parameters = inspect.signature(BehaviorPrimitives.pi0_nav_pick).parameters
    assert set(parameters) == {
        "self",
        "instruction",
        "chunks",
        "current_object_visual_check",
    }
    for forbidden in (
        "role",
        "hand",
        "visual_hand_check",
        "selected_hand",
        "projection_id",
        "navigation_visual_check",
        "standoff_m",
        "max_travel_m",
    ):
        assert forbidden not in parameters

    visual_check = properties["current_object_visual_check"]
    assert visual_check["properties"]["camera"]["enum"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]
    serialized = repr(visual_check)
    assert "held_wrist" not in serialized
    assert "press_wrist" not in serialized


def test_official_success_only_reads_raw_done_success():
    source = inspect.getsource(BehaviorEnvFacade._latch_official_success)
    assert "_raw_success" in source
    for forbidden in ("primitive_success", "reward", "green", "visual_marker"):
        assert forbidden not in source


def test_internal_success_helpers_reject_non_boolean_truthy_values():
    for helper in (tools_official_task_success, planner_official_task_success):
        assert helper({"done": {"success": True}}) is True
        assert helper({"done": {"success": np.bool_(True)}}) is True
        for value in (1, "true", [True], {"value": True}, object()):
            assert helper({"done": {"success": value}}) is False


def test_dual_attachment_at_step_23_does_not_interrupt_admitted_chunk():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    calls = {"count": 0}

    def step_env(_action, *, need_obs):
        assert need_obs is True
        calls["count"] += 1
        return (
            object(),
            np.array([0.0]),
            np.array([False]),
            np.array([False]),
            [{"done": {"success": False}}],
        )

    wrapped = {
        "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["turn on the radio"],
    }
    facade._env = SimpleNamespace(
        _direct_process=SimpleNamespace(step_env=step_env),
        _wrap_obs=lambda _obs: wrapped,
    )
    facade._done = False
    facade._last_observation = {
        "main_images": wrapped["main_images"][0],
        "wrist_images": wrapped["wrist_images"][0],
        "states": wrapped["states"][0],
        "task_descriptions": wrapped["task_descriptions"][0],
    }
    facade._last_info = {"done": {"success": False}}
    facade._meta = {"max_episode_steps": 128}
    facade._env_steps = 0
    facade._planner_video_interval_steps = 4
    facade._gripper_latch = {"left": 1.0, "right": 1.0}
    facade._official_success_latched = False
    facade._official_success_receipt = None
    facade._controller_state = "vla"
    facade._action_source = "pi0_vla"
    facade._vla_actions_enabled = True
    facade._motion_in_flight = True
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda *_args, **_kwargs: None
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 2 if calls["count"] >= 23 else 0,
        "hands": ["left", "right"] if calls["count"] >= 23 else [],
        "identity_conflict": False,
        "by_hand": {
            "left": {"attached": calls["count"] >= 23},
            "right": {"attached": calls["count"] >= 23},
        },
    }
    facade._physical_gripper_opening = lambda _hand: 0.0

    result = facade._step_action_chunk(
        np.zeros((32, 23), dtype=np.float32),
        observe_final=True,
        pi0_nav_pick=True,
    )

    assert calls["count"] == 32
    monitor = result[4]["_rpent"]["pi0_nav_pick_monitor"]
    assert monitor["executed_steps"] == 32
    assert monitor["local_grasp_success"] is True
    capability = monitor["capability"]
    assert capability["attachments"] == {
        "available": True,
        "count": 2,
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": True},
        },
        "conflict": False,
    }
    assert "dynamic_role_availability" not in capability
    assert "semantic_role" not in capability


def test_pi0_guard_accepts_two_distinct_attachment_identities_with_fresh_review():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._reset_completed = True
    facade._motion_in_flight = False
    facade._official_success_latched = False
    facade._last_info = {"done": {"success": False}}
    facade._terminal_failure_receipt = None
    facade._controller_state = "planner"
    facade._attempt_index = 1
    facade._attempt_nonce = "attempt"
    facade._env_steps = 7
    facade._pi0_nav_pick_is_receipt_disabled = lambda: False
    left_root = SimpleNamespace(prim_path="/World/left-can")
    right_root = SimpleNamespace(prim_path="/World/right-bin")
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 2,
        "hands": ["left", "right"],
        "identity_conflict": False,
        "attached_objects": {
            "left": {"left_eef_link": left_root},
            "right": {"right_eef_link": right_root},
        },
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": True},
        },
    }
    visual_check = {
        "camera": "head",
        "frame_id": "head:7:fresh",
        "assessment": "current_task_object_configuration_reviewed",
    }
    facade._current_object_visual_authorization = lambda check, *, invocation_id: {
        "invocation_id": invocation_id,
        "frame_id": check["frame_id"],
    }

    guarded = facade.guard_tool_call(
        name="pi0_nav_pick",
        input_dict={"current_object_visual_check": visual_check},
    )

    assert guarded["primitive_success"] is True
    assert guarded["failed_preconditions"] == []
    assert "ambiguous_attachment" not in guarded["failed_preconditions"]

    facade._attachment_runtime_facts = lambda: {
        "available": False,
        "attachment_count": 2,
        "hands": ["left", "right"],
        "identity_conflict": True,
        "attached_objects": {},
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": True},
        },
    }
    conflict = facade.guard_tool_call(
        name="pi0_nav_pick",
        input_dict={"current_object_visual_check": visual_check},
    )
    assert conflict["primitive_success"] is False
    assert conflict["failed_preconditions"] == ["attachment_identity_conflict"]
