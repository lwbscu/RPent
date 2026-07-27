from __future__ import annotations

import hashlib
import inspect
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import _ENV_RPC_METHODS, BehaviorEnvFacade
from robots.behavior.planner_executor import PlannerExecutor
from robots.behavior.schemas import (
    CLOSE_SPEC,
    MOVE_TO_SPEC,
    OBSERVE_SPEC,
    OPEN_SPEC,
    PRESS_SPEC,
    ROTATE_WRIST_RUNTIME_TIMEOUT_S,
    ROTATE_WRIST_SPEC,
)
from robots.behavior.tools import BehaviorPrimitives


def _facade() -> BehaviorEnvFacade:
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._run_nonce = "run"
    facade._attempt_nonce = "attempt"
    facade._attempt_index = 1
    facade._env_steps = 7
    facade._motion_in_flight = False
    facade._official_success_latched = False
    facade._controller_state = "planner"
    facade._reset_completed = True
    facade._last_info = {"done": {"success": False}}
    facade._terminal_failure_receipt = None
    facade._gripper_latch = {"left": -1.0, "right": 1.0}
    facade._held_closure_receipts = {}
    facade._projection_receipts = {}
    facade._consumed_projection_receipts = set()
    facade._public_observed_frame_ids = {"head:7:visual-selection"}
    facade._latest_public_head_frame_id = "head:7:visual-selection"
    facade._public_capture_sequence = 0
    facade._latest_unconsumed_public_capture_receipt = None
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: SimpleNamespace(
            frame_id=frame_id,
            step_index=7,
            capture_group_id="capture:7:visual-selection",
        )
    )
    facade._held_rotate_target_surface_review = None
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_eef_pose=lambda hand: (
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
        )
    )
    facade._attachment_runtime_facts = lambda: {
        "attachment_count": 1,
        "hands": ["left"],
        "identity_conflict": False,
        "available": True,
        "attached_objects": {"left": {"left_eef_link": object()}},
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": False},
        },
    }
    facade._physical_gripper_opening = lambda hand: 0.0 if hand == "left" else 1.0
    return facade


def _visual_hand_check(hand: str, *, frame_id: str = "head:7:visual-selection"):
    return {
        "camera": "head",
        "frame_id": frame_id,
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _depth_probe(
    *,
    frame_id: str = "head:7:depth-target",
    u: int = 3,
    v: int = 2,
    depth_window_px: int = 7,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "u": u,
        "v": v,
        "depth_window_px": depth_window_px,
        "assessment": "target_point_visually_confirmed",
    }


def _successful_isolation(hand: str, *, mode: str) -> dict[str, object]:
    return {
        "single_arm_isolation": {
            "available": True,
            "ok": True,
            "selected_hand": hand,
            "mode": mode,
            "stop_reason": "single_arm_isolation_verified",
        }
    }


def _capture_receipt(
    facade: BehaviorEnvFacade,
    frame: SimpleNamespace,
    *,
    camera: str = "head",
    image_bytes: bytes = b"reviewed-rgb",
) -> dict:
    facade._public_capture_sequence += 1
    receipt = facade._seal_attempt_receipt(
        {
            "kind": "public_observe_capture",
            "requested_camera": camera,
            "resolved_camera": camera,
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "env_step": facade._env_steps,
            "capture_sequence": facade._public_capture_sequence,
            "rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
    )
    facade._latest_unconsumed_public_capture_receipt = receipt
    return receipt


def test_runtime_rpc_surface_contains_only_neutral_primitives():
    assert {
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "prepare_vla_invocation",
        "pi0_nav_pick_chunk_step",
    } <= _ENV_RPC_METHODS
    forbidden = {
        "inspect_post_pick_state",
        "inspect_toggle_geometry",
        "prepress_move_to",
        "prepress_rotate_wrist",
        "post_pick_close_press_gripper",
        "post_pick_recenter_held_button",
        "post_pick_direct_finger_toggle",
        "post_success_hold_frames",
    }
    assert _ENV_RPC_METHODS.isdisjoint(forbidden)
    assert all(not hasattr(BehaviorEnvFacade, name) for name in forbidden)


def test_sanitized_capability_has_per_hand_state_but_no_privileged_geometry():
    capability = _facade()._sanitized_capability_summary()
    assert capability["attachments"] == {
        "available": True,
        "count": 1,
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": False},
        },
        "conflict": False,
    }
    assert capability["gripper_state"] == {
        "left": "closed",
        "right": "open",
    }
    assert "held_object" not in capability
    assert "dynamic_role_availability" not in capability
    forbidden = {
        "xyz",
        "pose",
        "air_gap_m",
        "attachment",
        "contact_count",
        "marker",
        "toggle",
    }
    assert forbidden.isdisjoint(capability)


@pytest.mark.parametrize("held_hand", ["left", "right"])
def test_held_large_object_is_effectively_closed_by_stable_attachment_receipt(
    held_hand,
):
    other_hand = "right" if held_hand == "left" else "left"
    eef_link = f"{held_hand}_eef_link"
    root = SimpleNamespace(prim_path="/World/task_object/root")
    attached = {eef_link: root}
    getter_calls = []

    def get_attached_object(hand):
        getter_calls.append(hand)
        return attached if hand == held_hand else None

    facade = _facade()
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(get_attached_object=get_attached_object)
    )
    facade._attachment_runtime_facts = (
        BehaviorEnvFacade._attachment_runtime_facts.__get__(facade)
    )
    facade._gripper_latch = {held_hand: -1.0, other_hand: -1.0}
    facade._physical_gripper_opening = lambda _hand: 0.049
    facade._held_closure_receipts = {
        held_hand: {
            "schema_version": 1,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "attempt_index": 1,
            "hand": held_hand,
            "expected_attachment": attached,
            "confirmed_env_step": 6,
            "close_latch": -1.0,
            "attachment_endpoint_held_steps": 10,
        }
    }

    capability = facade._sanitized_capability_summary()

    assert getter_calls == ["left", "right"]
    assert capability["gripper_state"] == {
        held_hand: "closed",
        other_hand: "open",
    }
    assert capability["attachments"]["by_hand"][held_hand]["attached"] is True
    assert capability["attachments"]["by_hand"][other_hand]["attached"] is False


@pytest.mark.parametrize(
    "mutation",
    ["too_few_steps", "wrong_attempt", "latch_reopened", "wrong_root", "lost"],
)
def test_held_effective_close_receipt_fails_closed_and_is_invalidated(mutation):
    expected_root = SimpleNamespace(prim_path="/World/task_object/root")
    current_root = expected_root
    attached = {"left_eef_link": current_root}
    receipt = {
        "schema_version": 1,
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "attempt_index": 1,
        "hand": "left",
        "expected_attachment": {"left_eef_link": expected_root},
        "confirmed_env_step": 6,
        "close_latch": -1.0,
        "attachment_endpoint_held_steps": 10,
    }
    latch = -1.0
    if mutation == "too_few_steps":
        receipt["attachment_endpoint_held_steps"] = 9
    elif mutation == "wrong_attempt":
        receipt["attempt_nonce"] = "old-attempt"
    elif mutation == "latch_reopened":
        latch = 1.0
    elif mutation == "wrong_root":
        attached = {
            "left_eef_link": SimpleNamespace(prim_path="/World/different_object/root")
        }
    elif mutation == "lost":
        attached = None

    facade = _facade()
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_attached_object=lambda hand: attached if hand == "left" else None
        )
    )
    facade._attachment_runtime_facts = (
        BehaviorEnvFacade._attachment_runtime_facts.__get__(facade)
    )
    facade._gripper_latch = {"left": latch, "right": 1.0}
    facade._physical_gripper_opening = lambda _hand: 0.049
    facade._held_closure_receipts = {"left": receipt}

    capability = facade._sanitized_capability_summary()

    if mutation == "lost":
        assert capability["attachments"]["count"] == 0
        assert capability["attachments"]["by_hand"]["left"]["attached"] is False
    else:
        assert capability["attachments"]["count"] == 1
        assert capability["attachments"]["by_hand"]["left"]["attached"] is True
    assert capability["gripper_state"]["left"] == "open"
    assert facade._held_closure_receipts == {}


def test_stable_held_receipt_never_satisfies_press_gripper_guard():
    root = SimpleNamespace(prim_path="/World/task_object/root")
    attached = {"left_eef_link": root}
    facade = _facade()
    facade._attachment_runtime_facts = lambda: {
        "attachment_count": 1,
        "hands": ["left"],
        "identity_conflict": False,
        "available": True,
        "attached_objects": {"left": attached},
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": False},
        },
    }
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    facade._physical_gripper_opening = lambda _hand: 0.049
    facade._held_closure_receipts = {
        "left": {
            "schema_version": 1,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "attempt_index": 1,
            "hand": "left",
            "expected_attachment": attached,
            "confirmed_env_step": 6,
            "close_latch": -1.0,
            "attachment_endpoint_held_steps": 10,
        }
    }

    guard = facade.guard_tool_call(
        name="press",
        input_dict={
            "hand": "right",
            "visual_hand_check": _visual_hand_check("right"),
        },
    )

    assert guard["primitive_success"] is False
    assert "closed_gripper_required" in guard["failed_preconditions"]


def test_press_requires_both_physical_closure_and_actual_close_latch():
    facade = _facade()
    facade._physical_gripper_opening = lambda hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": 1.0}

    capability = facade._sanitized_capability_summary()
    guard = facade.guard_tool_call(
        name="press",
        input_dict={
            "hand": "right",
            "visual_hand_check": _visual_hand_check("right"),
        },
    )

    assert capability["gripper_state"]["right"] == "closed"
    assert guard["primitive_success"] is False
    assert "closed_gripper_required" in guard["failed_preconditions"]

    facade._gripper_latch["right"] = -1.0
    capability = facade._sanitized_capability_summary()
    guard = facade.guard_tool_call(
        name="press",
        input_dict={
            "hand": "right",
            "visual_hand_check": _visual_hand_check("right"),
        },
    )

    assert capability["gripper_state"]["right"] == "closed"
    assert guard["primitive_success"] is True


def test_press_execution_rechecks_actual_close_latch_before_motion():
    facade = _facade()
    facade._physical_gripper_opening = lambda _hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": 1.0}
    facade._projection_receipts["projection-1"] = {
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "env_step": 7,
        "projection_id": "projection-1",
        "world_point": [0.1, 0.2, 0.3],
        "camera_facing_normal": [0.0, 0.0, 1.0],
    }

    with pytest.raises(RuntimeError, match="physical opening.*actual close latch"):
        facade.press(
            hand="right",
            projection_id="projection-1",
            travel_m=0.03,
            visual_hand_check=_visual_hand_check("right"),
        )


def test_successful_held_open_invalidates_stable_closure_receipt():
    root = SimpleNamespace(prim_path="/World/task_object/root")
    attached = {"left_eef_link": root}

    attachment_state = {"attached": attached}

    class Planner:
        backend = SimpleNamespace()

        def _gripper_command(self, hand, **kwargs):
            assert hand == "left"
            assert kwargs["opening"] == 1.0
            attachment_state["attached"] = None
            return {
                "primitive_success": True,
                "stop_reason": "gripper_commanded",
                "recoverable": True,
                "metrics": _successful_isolation("left", mode="gripper_only"),
                "diagnostics": {},
            }

    facade = _facade()
    facade._attachment_runtime_facts = lambda: {
        "attachment_count": int(attachment_state["attached"] is not None),
        "hands": ["left"] if attachment_state["attached"] is not None else [],
        "identity_conflict": False,
        "available": True,
        "attached_objects": (
            {"left": attachment_state["attached"]}
            if attachment_state["attached"] is not None
            else {}
        ),
        "by_hand": {
            "left": {"attached": attachment_state["attached"] is not None},
            "right": {"attached": False},
        },
    }
    facade._planner = Planner()
    facade._held_closure_receipts = {
        "left": {
            "schema_version": 1,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "attempt_index": 1,
            "hand": "left",
            "expected_attachment": attached,
            "confirmed_env_step": 6,
            "close_latch": -1.0,
            "attachment_endpoint_held_steps": 10,
        }
    }

    result = facade.open(
        hand="left",
        visual_hand_check=_visual_hand_check("left"),
    )

    assert result["primitive_success"] is True
    assert facade._held_closure_receipts == {}


@pytest.mark.parametrize("held_hand", ["left", "right"])
def test_successful_held_close_mints_run_bound_stable_attachment_receipt(
    held_hand,
):
    other_hand = "right" if held_hand == "left" else "left"
    root = SimpleNamespace(prim_path="/World/task_object/root")
    attached = {f"{held_hand}_eef_link": root}
    facade = _facade()

    class Planner:
        backend = SimpleNamespace(
            get_attached_object=lambda hand: attached if hand == held_hand else None
        )

        def _gripper_command(self, hand, **kwargs):
            assert hand == held_hand
            assert kwargs["opening"] == 0.0
            assert kwargs["require_attachment"] is True
            assert kwargs["hold_steps_required"] == 10
            assert kwargs["expected_attachment"] is attached
            facade._gripper_latch[hand] = -1.0
            facade._env_steps += 12
            return {
                "primitive_success": True,
                "stop_reason": "gripper_commanded",
                "recoverable": True,
                "metrics": {
                    "attachment_endpoint_held_steps": 10,
                    "attachment_confirmation_steps": 12,
                    **_successful_isolation(held_hand, mode="gripper_only"),
                },
                "diagnostics": {},
            }

    facade._planner = Planner()
    facade._attachment_runtime_facts = (
        BehaviorEnvFacade._attachment_runtime_facts.__get__(facade)
    )
    facade._gripper_latch = {held_hand: 1.0, other_hand: 1.0}
    facade._physical_gripper_opening = lambda _hand: 0.049

    result = facade.close(
        hand=held_hand,
        visual_hand_check=_visual_hand_check(held_hand),
    )

    receipt = facade._held_closure_receipts.get(held_hand)
    assert result["primitive_success"] is True
    assert isinstance(receipt, dict)
    assert receipt["run_nonce"] == "run"
    assert receipt["attempt_nonce"] == "attempt"
    assert receipt["attempt_index"] == 1
    assert receipt["hand"] == held_hand
    assert receipt["close_latch"] == pytest.approx(-1.0)
    assert receipt["attachment_endpoint_held_steps"] == 10
    assert result["capability"]["gripper_state"][held_hand] == "closed"
    assert result["capability"]["attachments"]["by_hand"][held_hand] == {
        "attached": True
    }


def test_two_attachments_are_publicly_valid_without_exposing_identity():
    left_root = SimpleNamespace(prim_path="/World/task_object/root")
    right_root = SimpleNamespace(prim_path="/World/other_object/root")
    left_attached = {"left_eef_link": left_root}
    right_attached = {"right_eef_link": right_root}
    facade = _facade()
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_attached_object=lambda hand: (
                left_attached if hand == "left" else right_attached
            )
        )
    )
    facade._attachment_runtime_facts = (
        BehaviorEnvFacade._attachment_runtime_facts.__get__(facade)
    )
    facade._held_closure_receipts = {
        "left": {
            "schema_version": 1,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "attempt_index": 1,
            "hand": "left",
            "expected_attachment": left_attached,
            "confirmed_env_step": 6,
            "close_latch": -1.0,
            "attachment_endpoint_held_steps": 10,
        }
    }

    capability = facade._sanitized_capability_summary()

    assert capability["attachments"] == {
        "available": True,
        "count": 2,
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": True},
        },
        "conflict": False,
    }
    serialized = json.dumps(capability, sort_keys=True)
    assert "/World/" not in serialized


def test_all_analytic_public_schemas_accept_only_explicit_physical_hands():
    for spec in (MOVE_TO_SPEC, ROTATE_WRIST_SPEC, CLOSE_SPEC, OPEN_SPEC, PRESS_SPEC):
        hand_schema = spec["input_schema"]["properties"]["hand"]
        assert hand_schema["enum"] == ["left", "right"]
        assert "role" not in spec["input_schema"]["properties"]
        assert "visual_hand_check" in spec["input_schema"]["required"]


def test_terminal_visual_failure_is_frame_bound_sealed_and_frozen(tmp_path):
    class FrameCache:
        ttl_s = 60.0

        def __init__(self, evidence_step=7):
            self.evidence_step = evidence_step

        def get_current(self, camera, frame_id):
            expected = {
                "head": "head:evidence",
                "left_wrist": "left_wrist:fresh",
                "right_wrist": "right_wrist:fresh",
            }[camera]
            if frame_id not in {"head:evidence", expected}:
                raise RuntimeError("unexpected frame")
            return SimpleNamespace(
                frame_id=frame_id,
                step_index=(self.evidence_step if frame_id == "head:evidence" else 7),
                capture_group_id=(
                    "evidence-capture"
                    if frame_id == "head:evidence"
                    else "evidence-capture"
                ),
            )

        @staticmethod
        def observe_payload(camera):
            return {
                "frame_id": (
                    "head:evidence" if camera == "head" else f"{camera}:fresh"
                ),
                "capture_group": {"id": "evidence-capture"},
                "_image_bytes": b"\x89PNG\r\n\x1a\nrgb-" + camera.encode(),
                "_depth_image_bytes": b"\x89PNG\r\n\x1a\ndepth-" + camera.encode(),
            }

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._run_nonce = "run"
    facade._attempt_nonce = "attempt"
    facade._attempt_index = 1
    facade._env_steps = 7
    facade._output_dir = tmp_path
    facade._frame_cache = FrameCache()
    facade._public_observed_frame_ids = {"head:evidence"}
    facade._visual_checkpoint_counter = 0
    facade._last_info = {"done": {"success": False}}
    facade._terminal_failure_receipt = None
    facade._terminal_failure_receipt_path = tmp_path / "terminal_failure_receipt.json"
    facade._official_success_latched = False
    facade._motion_frozen = False
    facade._controller_state = "planner"
    facade._action_source = "planner"
    facade._vla_actions_enabled = False
    facade._done = False
    facade._video_error = None
    facade._video_path = tmp_path / "episode.mp4"
    facade._video_sealed = False
    facade._refresh_observation_without_step = lambda: None
    facade._finalize_video_segment = lambda: None

    result = facade.save_robot_state_checkpoint(
        semantic_label="terminal visual evidence",
        terminal_failure={
            "condition": "radio_tipped_flat",
            "cause": "dropped_out_of_gripper",
            "camera": "head",
            "frame_id": "head:evidence",
        },
    )

    assert result["_finish"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "radio_tipped_flat"
    assert result["runner_termination_reason"] == "visual_radio_tipped_flat"
    assert result["terminal_failure_receipt"]["frame_id"] == "head:evidence"
    assert facade._motion_frozen is True
    assert facade._controller_state == "frozen"
    assert facade._action_source == "frozen"
    assert facade._done is True
    assert facade._terminal_failure_receipt_path.is_file()

    unpublished = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    unpublished._env_steps = 7
    unpublished._frame_cache = FrameCache()
    unpublished._public_observed_frame_ids = set()
    with pytest.raises(RuntimeError, match="public observe result"):
        unpublished.save_robot_state_checkpoint(
            terminal_failure={
                "condition": "radio_tipped_flat",
                "cause": "knocked_over_by_robot_hand",
                "camera": "head",
                "frame_id": "head:evidence",
            }
        )

    stale = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    stale._env_steps = 7
    stale._frame_cache = FrameCache(evidence_step=6)
    stale._public_observed_frame_ids = {"head:evidence"}
    with pytest.raises(RuntimeError, match="not from the current env step"):
        stale.save_robot_state_checkpoint(
            terminal_failure={
                "condition": "radio_tipped_flat",
                "cause": "knocked_over_by_robot_hand",
                "camera": "head",
                "frame_id": "head:evidence",
            }
        )


def test_projection_receipt_is_attempt_and_step_bound_and_replay_safe():
    facade = _facade()
    receipt = {
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "env_step": 7,
        "projection_id": "projection-1",
        "world_point": [0.1, 0.2, 0.3],
        "camera_facing_normal": [0.0, 0.0, 1.0],
    }
    facade._projection_receipts["projection-1"] = receipt
    assert facade._projection_receipt_is_fresh(receipt)
    target, receipt_id = facade._motion_target(
        hand="left",
        target={"projection_id": "projection-1", "standoff_m": 0.05},
        max_travel_m=0.1,
    )
    assert np.allclose(target, [0.1, 0.2, 0.35])
    assert receipt_id == "projection-1"
    facade._consumed_projection_receipts.add("projection-1")
    try:
        facade._motion_target(
            hand="left",
            target={"projection_id": "projection-1", "standoff_m": 0.05},
            max_travel_m=0.1,
        )
    except RuntimeError as exc:
        assert "consumed" in str(exc)
    else:
        raise AssertionError("consumed projection was accepted")


def test_production_source_has_no_task_specific_target_generation():
    source = inspect.getsource(BehaviorEnvFacade)
    for forbidden in (
        "ToggledOn",
        "visual_marker",
        "RADIO_LOCAL_BUTTON",
        "_resolve_handoff_targets",
        "clearance_first_required",
        "pi0_nav_pick_required_first",
    ):
        assert forbidden not in source


def test_public_results_strip_ordering_and_stage_advice_recursively():
    stripped = BehaviorEnvFacade._strip_flow_advice(
        {
            "primitive_success": True,
            "suggested_next_tool": "move_to",
            "metrics": {
                "reachability_stage": "legacy",
                "stages": {"first": {"suggested_next_tool": "press"}},
                "collision_margin_m": 0.01,
            },
        }
    )
    assert stripped == {
        "primitive_success": True,
        "metrics": {"collision_margin_m": 0.01},
    }


def test_navigation_is_public_while_legacy_pick_and_release_remain_absent():
    assert callable(getattr(PlannerExecutor, "navigate_to"))
    parameters = inspect.signature(PlannerExecutor.navigate_to).parameters
    assert set(parameters) == {
        "self",
        "target_xyz",
        "relative_motion",
        "standoff_m",
        "max_travel_m",
        "timeout_s",
    }
    assert {"hand", "role", "projection_id"}.isdisjoint(parameters)
    assert not hasattr(PlannerExecutor, "pick")
    assert not hasattr(PlannerExecutor, "release")


def test_press_is_one_neutral_motion_and_never_closes_the_gripper():
    signature = inspect.signature(PlannerExecutor.press)
    assert "travel_m" in signature.parameters
    assert "approach_distance_m" not in signature.parameters
    assert "press_depth_m" not in signature.parameters
    assert "_gripper_command" not in inspect.getsource(PlannerExecutor.press)

    executor = PlannerExecutor.__new__(PlannerExecutor)
    executor._validated_timeout = lambda timeout: float(timeout)
    executor._remaining_s = lambda deadline: 10.0
    executor._task_success = lambda: False
    executor._persist_tool_artifact = lambda **kwargs: "tool.json"
    executor._capture_single_arm_isolation = lambda **_kwargs: {
        "selected_hand": "left",
        "mode": "arm_motion",
        "context_id": "press-isolation",
        "reference_origin": "composite_tool_call_start",
    }
    executor._move_to_composite_stage = lambda **kwargs: {
        "primitive_success": True,
        "metrics": {"precontact": True},
    }
    guarded_calls = []
    executor._guarded_incremental_move = lambda **kwargs: (
        guarded_calls.append(kwargs)
        or {
            "primitive_success": True,
            "stop_reason": "contact",
            "recoverable": True,
            "metrics": {},
            "diagnostics": {},
        }
    )
    executor._gripper_command = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("press must not close a gripper")
    )

    result = executor.press(
        hand="left",
        target_xyz=[0.1, 0.2, 0.3],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.02,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is True
    assert guarded_calls
    assert result["metrics"]["requested_travel_m"] == 0.02


def test_server_public_signatures_match_the_neutral_schema():
    assert list(inspect.signature(BehaviorEnvFacade.observe).parameters) == [
        "self",
        "camera",
        "frame_review",
        "depth_probe",
    ]
    assert list(inspect.signature(BehaviorEnvFacade.pixel_to_world).parameters) == [
        "self",
        "camera",
        "frame_id",
        "u",
        "v",
        "depth_window_px",
    ]
    move_parameters = inspect.signature(BehaviorEnvFacade.move_to).parameters
    assert set(move_parameters) == {
        "self",
        "hand",
        "target",
        "visual_hand_check",
        "position_tolerance_m",
        "max_travel_m",
        "timeout_s",
    }
    press_parameters = inspect.signature(BehaviorEnvFacade.press).parameters
    assert set(press_parameters) == {
        "self",
        "hand",
        "projection_id",
        "travel_m",
        "visual_hand_check",
        "timeout_s",
    }
    for name in ("close", "open"):
        parameters = inspect.signature(getattr(BehaviorEnvFacade, name)).parameters
        expected = {
            "self",
            "hand",
            "visual_hand_check",
            "timeout_s",
        }
        if name == "open":
            expected.add("release_visual_check")
        assert set(parameters) == expected


def test_public_defaults_and_required_parameters_match_across_all_layers():
    assert OBSERVE_SPEC["input_schema"]["required"] == ["camera"]
    for method in (
        BehaviorPrimitives.observe,
        BehaviorEnvClient.observe,
        BehaviorEnvFacade.observe,
    ):
        parameters = inspect.signature(method).parameters
        assert list(parameters) == ["self", "camera", "frame_review", "depth_probe"]
        assert parameters["camera"].default is inspect.Signature.empty
        assert parameters["frame_review"].default is None
        assert parameters["depth_probe"].default is None

    assert ROTATE_WRIST_SPEC["input_schema"]["properties"]["frame"]["default"] == "eef"
    analytic_methods = (
        (
            BehaviorPrimitives.move_to,
            BehaviorEnvClient.move_to,
            BehaviorEnvFacade.move_to,
        ),
        (
            BehaviorPrimitives.rotate_wrist,
            BehaviorEnvClient.rotate_wrist,
            BehaviorEnvFacade.rotate_wrist,
        ),
        (BehaviorPrimitives.close, BehaviorEnvClient.close, BehaviorEnvFacade.close),
        (BehaviorPrimitives.open, BehaviorEnvClient.open, BehaviorEnvFacade.open),
        (BehaviorPrimitives.press, BehaviorEnvClient.press, BehaviorEnvFacade.press),
    )
    for layer_methods in analytic_methods:
        for method in layer_methods:
            parameters = inspect.signature(method).parameters
            assert parameters["hand"].default is inspect.Signature.empty
            assert "role" not in parameters
            assert parameters["visual_hand_check"].default is inspect.Signature.empty
    for method in analytic_methods[1]:
        parameters = inspect.signature(method).parameters
        assert parameters["frame"].default == "eef"
        assert parameters["relative_axis_angle"].default is inspect.Signature.empty
    for method in (
        BehaviorPrimitives.open,
        BehaviorEnvClient.open,
        BehaviorEnvFacade.open,
    ):
        assert (
            inspect.signature(method).parameters["release_visual_check"].default is None
        )
    assert "release_visual_check" not in OPEN_SPEC["input_schema"]["required"]

    assert "travel_m" in PRESS_SPEC["input_schema"]["required"]
    for method in (
        BehaviorPrimitives.press,
        BehaviorEnvClient.press,
        BehaviorEnvFacade.press,
    ):
        parameters = inspect.signature(method).parameters
        assert parameters["travel_m"].default is inspect.Signature.empty
        assert parameters["hand"].default is inspect.Signature.empty
        assert parameters["projection_id"].default is inspect.Signature.empty


def test_observe_wrapper_forwards_capture_review_or_exact_depth_probe(tmp_path):
    calls: list[dict] = []
    env = SimpleNamespace(
        total_env_steps=7,
        observe=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "observed",
                "official_success_source": 'info["done"]["success"]',
            }
        ),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)

    primitives.observe(camera="head")
    primitives.observe(
        camera="left_wrist",
        frame_review={
            "frame_id": "left_wrist:7:fresh",
            "assessment": "target_bearing_surface_confirmed",
        },
    )
    primitives.observe(
        camera="right_wrist",
        depth_probe={
            **_depth_probe(frame_id="right_wrist:7:fresh"),
            "u": np.int64(4),
            "v": np.int32(5),
            "depth_window_px": np.int64(9),
        },
    )

    assert calls == [
        {"camera": "head"},
        {
            "camera": "left_wrist",
            "frame_review": {
                "frame_id": "left_wrist:7:fresh",
                "assessment": "target_bearing_surface_confirmed",
            },
        },
        {
            "camera": "right_wrist",
            "depth_probe": {
                "frame_id": "right_wrist:7:fresh",
                "u": 4,
                "v": 5,
                "depth_window_px": 9,
                "assessment": "target_point_visually_confirmed",
            },
        },
    ]
    for invalid in (
        {"frame_id": "head:7:fresh"},
        {"assessment": "side_or_indeterminate"},
        {
            "frame_id": "",
            "assessment": "side_or_indeterminate",
        },
        {
            "frame_id": "head:7:fresh",
            "assessment": "unsupported",
        },
        {
            "frame_id": "head:7:fresh",
            "assessment": "opposite_surface_confirmed",
            "extra": True,
        },
    ):
        with pytest.raises(ValueError):
            primitives.observe(camera="head", frame_review=invalid)
    for invalid in (
        {"frame_id": "head:7:fresh"},
        {key: value for key, value in _depth_probe().items() if key != "assessment"},
        {**_depth_probe(), "frame_id": ""},
        {**_depth_probe(), "u": True},
        {**_depth_probe(), "u": 1.5},
        {**_depth_probe(), "v": False},
        {**_depth_probe(), "depth_window_px": 0},
        {**_depth_probe(), "depth_window_px": 32},
        {**_depth_probe(), "depth_window_px": True},
        {**_depth_probe(), "assessment": "target_probably_visible"},
        {**_depth_probe(), "extra": True},
    ):
        with pytest.raises(ValueError):
            primitives.observe(camera="head", depth_probe=invalid)
    with pytest.raises(ValueError, match="mutually exclusive"):
        primitives.observe(
            camera="head",
            frame_review={
                "frame_id": "head:7:fresh",
                "assessment": "side_or_indeterminate",
            },
            depth_probe=_depth_probe(frame_id="head:7:fresh"),
        )
    assert len(calls) == 3


def test_navigate_to_wrapper_forwards_only_projection_bound_base_arguments(tmp_path):
    calls: list[dict] = []
    env = SimpleNamespace(
        total_env_steps=7,
        navigate_to=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "navigation_complete",
                "official_success_source": 'info["done"]["success"]',
            }
        ),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    visual_check = {
        "camera": "head",
        "frame_id": "head:7:navigation-target",
        "assessment": "navigation_target_visually_confirmed",
    }

    primitives.navigate_to(
        projection_id=" projection-current ",
        navigation_visual_check=visual_check,
        standoff_m=0.9,
        max_travel_m=1.2,
        timeout_s=250,
    )

    assert calls == [
        {
            "projection_id": "projection-current",
            "navigation_visual_check": visual_check,
            "standoff_m": 0.9,
            "max_travel_m": 1.2,
            "timeout_s": 250.0,
        }
    ]
    assert {
        "hand",
        "role",
        "target_xyz",
        "delta_xyz",
        "frame",
        "chunks",
        "max_chunks",
    }.isdisjoint(calls[0])

    for invalid_check in (
        None,
        {**visual_check, "camera": "left_wrist"},
        {**visual_check, "frame_id": ""},
        {**visual_check, "assessment": "target_point_visually_confirmed"},
        {**visual_check, "selected_hand": "left"},
    ):
        with pytest.raises(ValueError):
            primitives.navigate_to(
                projection_id="projection-current",
                navigation_visual_check=invalid_check,
            )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "relative_motion",
    [
        {"kind": "translation", "direction": "forward", "distance_m": 0.25},
        {"kind": "translation", "direction": "backward", "distance_m": 0.25},
        {"kind": "rotation", "direction": "left", "angle_deg": 90.0},
        {"kind": "rotation", "direction": "right", "angle_deg": 90.0},
    ],
)
def test_navigate_to_wrapper_forwards_explicit_relative_motion(
    tmp_path,
    relative_motion,
):
    calls: list[dict] = []
    env = SimpleNamespace(
        total_env_steps=7,
        navigate_to=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "reached",
                "official_success_source": 'info["done"]["success"]',
            }
        ),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)

    primitives.navigate_to(relative_motion=relative_motion, timeout_s=120)

    assert calls == [
        {
            "relative_motion": relative_motion,
            "timeout_s": 120.0,
        }
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        primitives.navigate_to(
            projection_id="projection-current",
            relative_motion=relative_motion,
        )


def test_observe_review_reuses_public_frame_without_refresh_or_persistence():
    facade = _facade()
    image_bytes = b"reviewed-rgb"
    frame = SimpleNamespace(
        timestamp_s=time.monotonic(),
        frame_id="head:7:reviewed",
        capture_group_id="capture-current",
        step_index=7,
        intrinsics=SimpleNamespace(width=8, height=6, fx=4.0, fy=4.0, cx=3.5, cy=2.5),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    facade._frame_cache = SimpleNamespace(
        ttl_s=60.0,
        get_current=lambda camera, frame_id: frame,
    )
    facade._public_observed_frame_ids = {frame.frame_id}
    capture_receipt = _capture_receipt(
        facade,
        frame,
        image_bytes=image_bytes,
    )
    facade._refresh_observation_without_step = lambda: pytest.fail(
        "review-only observe must not refresh sensors"
    )
    facade._persist_live_observation = lambda _payload: pytest.fail(
        "review-only observe must not persist a new capture"
    )
    planner_calls = []
    facade._planner = SimpleNamespace(
        observe=lambda camera: (
            planner_calls.append(camera)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "observed",
                "camera": camera,
                "frame_id": frame.frame_id,
                "capture_group": {
                    "id": frame.capture_group_id,
                    "sim_step": 7,
                    "cameras": ["head", "left_wrist", "right_wrist"],
                },
                "metrics": {"camera": camera},
                "_image_bytes": image_bytes,
            }
        )
    )
    review_calls = []
    facade._record_frame_review_cycle = lambda **kwargs: (
        review_calls.append(kwargs)
        or {
            "accepted": True,
            "qualifying_cycle": False,
            "completed_qualifying_cycles": 0,
            "pi0_nav_pick_disabled": False,
        }
    )

    result = facade.observe(
        "head",
        {
            "frame_id": frame.frame_id,
            "assessment": "side_or_indeterminate",
        },
    )

    assert planner_calls == ["head"]
    assert review_calls == [
        {
            "requested_camera": "head",
            "resolved_camera": "head",
            "frame": frame,
            "capture_receipt": capture_receipt,
            "assessment": "side_or_indeterminate",
        }
    ]
    assert result["frame_id"] == frame.frame_id
    assert result["frame_review"] == {
        "accepted": True,
        "qualifying_cycle": False,
        "completed_qualifying_cycles": 0,
        "pi0_nav_pick_disabled": False,
    }
    assert "cameras" not in result["capture_group"]


def test_observe_review_rejects_nonpublic_or_stale_frame_before_planner_read():
    for published, frame_step, message in (
        (False, 7, "public observe"),
        (True, 6, "current env step"),
    ):
        facade = _facade()
        frame = SimpleNamespace(
            frame_id="head:review",
            capture_group_id="capture-current",
            step_index=frame_step,
        )
        facade._frame_cache = SimpleNamespace(
            get_current=lambda camera, frame_id: frame,
        )
        facade._public_observed_frame_ids = {frame.frame_id} if published else set()
        _capture_receipt(facade, frame)
        facade._planner = SimpleNamespace(
            observe=lambda _camera: pytest.fail(
                "invalid review must not read the planner observation"
            )
        )

        with pytest.raises(RuntimeError, match=message):
            facade.observe(
                "head",
                {
                    "frame_id": frame.frame_id,
                    "assessment": "opposite_surface_confirmed",
                },
            )


def test_observe_and_projection_do_not_expose_resolved_physical_side():
    facade = _facade()
    image_bytes = b"fresh-rgb"
    frame = SimpleNamespace(
        timestamp_s=time.monotonic(),
        frame_id="frame-current",
        capture_group_id="capture-current",
        step_index=7,
        intrinsics=SimpleNamespace(width=8, height=6, fx=4.0, fy=4.0, cx=3.5, cy=2.5),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    facade._frame_cache = SimpleNamespace(
        ttl_s=60.0,
        latest=lambda camera: frame,
        get_current=lambda camera, frame_id: frame,
    )
    facade._refresh_observation_without_step = (
        lambda *, synchronize_hand_geometry=False: None
    )
    facade._planner = SimpleNamespace(
        observe=lambda camera: {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "observed",
            "camera": camera,
            "frame_id": frame.frame_id,
            "capture_group": {
                "id": frame.capture_group_id,
                "sim_step": 7,
                "cameras": ["head", "left_wrist", "right_wrist"],
            },
            "metrics": {"camera": camera},
            "_image_bytes": image_bytes,
        },
        pixel_to_world=lambda **kwargs: {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "projected",
            "metrics": {"camera": kwargs["camera"], "confidence": 0.9},
            "diagnostics": {
                "xyz": [0.1, 0.2, 0.3],
                "surface_normal": [0.0, 0.0, 1.0],
                "output_frame": "world",
            },
        },
    )

    observed = facade.observe("left_wrist")
    projected = facade.pixel_to_world(
        "left_wrist",
        "frame-current",
        3,
        2,
        3,
    )

    assert observed["camera"] == "left_wrist"
    assert observed["camera_metadata"]["camera"] == "left_wrist"
    assert "cameras" not in observed["capture_group"]
    assert projected["camera"] == "left_wrist"
    assert projected["frame_id"] == "frame-current"
    assert "projection_receipt" not in projected
    serialized = json.dumps(
        [observed, projected],
        sort_keys=True,
        default=lambda value: "<bytes>" if isinstance(value, bytes) else str(value),
    )
    for forbidden in (
        "resolved_camera",
        "resolved_hand",
    ):
        assert forbidden not in serialized


def test_projection_rejects_a_frame_not_returned_by_public_observe():
    facade = _facade()
    frame = SimpleNamespace(
        frame_id="private-frame",
        capture_group_id="capture-current",
        step_index=7,
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: frame,
    )
    facade._public_observed_frame_ids = set()
    facade._planner = SimpleNamespace(
        pixel_to_world=lambda **_kwargs: pytest.fail(
            "projection RPC must not run for a non-public frame"
        )
    )

    with pytest.raises(RuntimeError, match="public observe frame"):
        facade.pixel_to_world("head", "private-frame", 3, 2, 3)


def test_all_analytic_wrappers_require_and_forward_visual_hand_check(tmp_path):
    calls: list[tuple[str, dict]] = []

    def record(name):
        return lambda **kwargs: (
            calls.append((name, kwargs))
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "reached",
                "official_success_source": 'info["done"]["success"]',
            }
        )

    env = SimpleNamespace(
        total_env_steps=7,
        move_to=record("move_to"),
        rotate_wrist=record("rotate_wrist"),
        close=record("close"),
        open=record("open"),
        press=record("press"),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    left_check = _visual_hand_check("left")
    right_check = _visual_hand_check("right")

    primitives.move_to(
        hand="left",
        target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
        visual_hand_check=left_check,
    )
    primitives.rotate_wrist(
        hand="right",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.4],
        visual_hand_check=right_check,
    )
    primitives.close(hand="left", visual_hand_check=left_check)
    primitives.open(hand="right", visual_hand_check=right_check)
    primitives.press(
        hand="right",
        projection_id="projection-current",
        travel_m=0.03,
        visual_hand_check=right_check,
    )

    assert [name for name, _kwargs in calls] == [
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
    ]
    assert all(kwargs["visual_hand_check"] for _name, kwargs in calls)
    assert calls[0][1]["hand"] == "left"
    assert calls[1][1]["hand"] == "right"
    assert calls[2][1]["hand"] == "left"
    assert calls[3][1]["hand"] == "right"
    assert calls[4][1]["hand"] == "right"


def test_all_analytic_wrappers_reject_legacy_roles_without_rpc(tmp_path):
    env = SimpleNamespace(
        total_env_steps=7,
        move_to=lambda **_kwargs: pytest.fail("legacy request reached move_to"),
        rotate_wrist=lambda **_kwargs: pytest.fail("legacy request reached rotate"),
        close=lambda **_kwargs: pytest.fail("legacy request reached close"),
        open=lambda **_kwargs: pytest.fail("legacy request reached open"),
        press=lambda **_kwargs: pytest.fail("legacy request reached press"),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    check = _visual_hand_check("left")

    for legacy in ("held", "press"):
        invocations = (
            lambda legacy_hand=legacy: primitives.move_to(
                hand=legacy_hand,
                target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
                visual_hand_check=check,
            ),
            lambda legacy_hand=legacy: primitives.rotate_wrist(
                hand=legacy_hand,
                relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
                visual_hand_check=check,
            ),
            lambda legacy_hand=legacy: primitives.close(
                hand=legacy_hand, visual_hand_check=check
            ),
            lambda legacy_hand=legacy: primitives.open(
                hand=legacy_hand, visual_hand_check=check
            ),
            lambda legacy_hand=legacy: primitives.press(
                hand=legacy_hand,
                projection_id="projection-current",
                travel_m=0.03,
                visual_hand_check=check,
            ),
        )
        for invoke in invocations:
            with pytest.raises(ValueError, match="hand must be"):
                invoke()

    with pytest.raises(TypeError):
        primitives.close(role="held", visual_hand_check=check)


def test_open_wrapper_forwards_only_exact_optional_release_visual_check(tmp_path):
    calls: list[dict] = []
    env = SimpleNamespace(
        total_env_steps=7,
        open=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "opened",
                "official_success_source": 'info["done"]["success"]',
            }
        ),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    visual_check = _visual_hand_check("left")
    release_check = {
        "camera": "head",
        "frame_id": "head:7:release",
        "selected_hand": "left",
        "assessment": "attached_object_fully_inside_receptacle_opening",
    }

    primitives.open(
        hand="left",
        visual_hand_check=visual_check,
        release_visual_check=release_check,
    )

    assert calls == [
        {
            "hand": "left",
            "visual_hand_check": visual_check,
            "release_visual_check": release_check,
            "timeout_s": 30.0,
        }
    ]
    for invalid in (
        {**release_check, "camera": "left_wrist"},
        {**release_check, "selected_hand": "right"},
        {**release_check, "assessment": "object_probably_inside"},
        {**release_check, "extra": True},
    ):
        with pytest.raises(ValueError, match="release_visual_check|hand must equal"):
            primitives.open(
                hand="left",
                visual_hand_check=visual_check,
                release_visual_check=invalid,
            )
    assert len(calls) == 1


def test_all_analytic_wrappers_reject_missing_or_empty_visual_check_without_rpc(
    tmp_path,
):
    env = SimpleNamespace(
        total_env_steps=7,
        move_to=lambda **_kwargs: pytest.fail("invalid request reached move_to"),
        rotate_wrist=lambda **_kwargs: pytest.fail("invalid request reached rotate"),
        close=lambda **_kwargs: pytest.fail("invalid request reached close"),
        open=lambda **_kwargs: pytest.fail("invalid request reached open"),
        press=lambda **_kwargs: pytest.fail("invalid request reached press"),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    calls = (
        lambda check: primitives.move_to(
            hand="left",
            target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
            visual_hand_check=check,
        ),
        lambda check: primitives.rotate_wrist(
            hand="right",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.4],
            visual_hand_check=check,
        ),
        lambda check: primitives.close(
            hand="left",
            visual_hand_check=check,
        ),
        lambda check: primitives.open(
            hand="right",
            visual_hand_check=check,
        ),
        lambda check: primitives.press(
            hand="right",
            projection_id="projection-current",
            travel_m=0.03,
            visual_hand_check=check,
        ),
    )
    for invoke in calls:
        for missing in (None, {}):
            with pytest.raises(ValueError, match="visual_hand_check"):
                invoke(missing)


def test_rotate_wrist_transport_retains_runtime_owned_deadline(tmp_path):
    calls: list[dict] = []
    env = SimpleNamespace(
        total_env_steps=7,
        rotate_wrist=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "reached",
                "official_success_source": 'info["done"]["success"]',
            }
        ),
    )
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    visual_check = _visual_hand_check("left")
    result = primitives.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.4],
        visual_hand_check=visual_check,
    )

    assert result["primitive_success"] is True
    assert calls == [
        {
            "hand": "left",
            "relative_axis_angle": [0.0, 1.0, 0.0, 0.4],
            "frame": "eef",
            "visual_hand_check": visual_check,
        }
    ]


def test_env_client_rotate_wrist_transports_visual_hand_check():
    calls: list[dict] = []
    client = BehaviorEnvClient.__new__(BehaviorEnvClient)
    client._client = SimpleNamespace(
        call=lambda method, **kwargs: calls.append({"method": method, **kwargs}) or {}
    )
    client.total_env_steps = 0
    client.episode_done = False
    visual_check = {
        "camera": "head",
        "frame_id": "head:current",
        "selected_hand": "right",
        "assessment": "selected_hand_visually_confirmed",
    }

    client.rotate_wrist(
        hand="right",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.4],
        frame="world",
        visual_hand_check=visual_check,
    )

    assert calls == [
        {
            "method": "env.rotate_wrist",
            "args": (),
            "kwargs": {
                "hand": "right",
                "relative_axis_angle": [0.0, 0.0, 1.0, 0.4],
                "frame": "world",
                "visual_hand_check": visual_check,
            },
            "timeout_s": ROTATE_WRIST_RUNTIME_TIMEOUT_S + 60.0,
        }
    ]


def test_pi0_nav_pick_layers_do_not_gain_hand_routing_parameters():
    for method in (
        BehaviorPrimitives.pi0_nav_pick,
        BehaviorEnvClient.pi0_nav_pick_chunk_step,
    ):
        parameters = inspect.signature(method).parameters
        assert {
            "role",
            "hand",
            "visual_hand_check",
            "projection_id",
            "navigation_visual_check",
            "standoff_m",
            "max_travel_m",
        }.isdisjoint(parameters)
