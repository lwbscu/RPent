import inspect
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import CameraGeometryError, CameraIntrinsics
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.planner_executor import EEF_LINK_BY_HAND, RealCuroboBackend
from robots.behavior.prepress import (
    AMBIGUOUS_FACE_CLASS,
    BUTTON_FACE_CLASS,
    CLEAR_SLOTTED_BACK_FACE_CLASS,
    POSITIVE_SIGNATURE_FIELDS,
    SIDE_PORT_FACE_CLASS,
    authorize_prepress_motion,
    evaluate_geometry,
    generate_button_goal_pose_candidates,
    generate_press_staging_pose_candidates,
    pose_matrix_xyzw,
    quat_between_vectors_xyzw,
    quat_rotate_xyzw,
    upright_radio_orientation_xyzw,
    validate_button_declaration,
)
from robots.behavior.schemas import PI0_NAV_PICK_VLA_MODE


def _signature():
    return dict.fromkeys(POSITIVE_SIGNATURE_FIELDS, True)


def _z_quat(degrees: float) -> list[float]:
    half = np.deg2rad(degrees) * 0.5
    return [0.0, 0.0, float(np.sin(half)), float(np.cos(half))]


def _held_button_goal(toward_robot_m: float = 0.10) -> dict:
    return {
        "kind": "held_button_alignment",
        "toward_robot_m": toward_robot_m,
        "head_view": "side",
        "face_toward": "press",
        "minimum_table_clearance_m": 0.12,
        "position_slack_m": 0.04,
        "candidate_budget": 12,
    }


def _press_button_goal() -> dict:
    return {
        "kind": "press_staging",
        "projection_id": "button_projection_test",
        "standoff_m": 0.055,
        "candidate_budget": 8,
    }


def _motion_guard_facade(
    tmp_path,
    *,
    face_class: str,
    held_hand: str = "right",
    press_hand: str = "left",
    gate_camera: str = "head",
):
    """Build a no-physics facade that exposes whether planning was reached."""

    calls = []
    root = SimpleNamespace(prim_path="/World/radio_89/root")
    q = np.zeros(24, dtype=np.float64)

    class Robot:
        base_idx = np.arange(0, 6)
        trunk_control_idx = np.arange(6, 10)
        arm_control_idx = {
            "right": np.arange(10, 17),
            "left": np.arange(17, 24),
        }

        @staticmethod
        def get_joint_positions():
            return q.copy()

    class Backend:
        @staticmethod
        def get_attached_object(hand):
            assert hand == held_hand
            return {EEF_LINK_BY_HAND[hand]: root}

        @staticmethod
        def plan_prepress_arm_trajectory(**kwargs):
            calls.append(("plan", kwargs))
            return {
                "ok": True,
                "joint_trajectory": q.reshape(1, -1),
                "metrics": {},
            }

        @staticmethod
        def _generator(*, kind, hand):
            calls.append(("generator", kind, hand))
            return object()

        @staticmethod
        def _check_q_trajectory_collisions(
            _generator,
            selected_q,
            *,
            attached_obj,
            **_kwargs,
        ):
            calls.append(
                (
                    "collision",
                    np.asarray(selected_q).copy(),
                    dict(attached_obj),
                )
            )
            return {
                "available": True,
                "colliding": False,
                "checked_waypoints": int(len(selected_q)),
            }

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 17
    facade._last_info = {"done": {"success": False}}
    facade._base_controller_mode = "position"
    facade._output_dir = tmp_path
    facade._prepress_round = 0
    facade._prepress_coarse_flip_used = False
    facade._prepress_context = {
        "held_hand": held_hand,
        "press_hand": press_hand,
        "transform_context": {
            "env_step": 17,
            "held_hand": held_hand,
            "press_hand": press_hand,
            "held_eef_pose_world": {
                "position": [0.0, 0.0, 0.60],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "press_eef_pose_world": {
                "position": [0.0, 0.0, 0.60],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }
    face_classes = {
        "clear_slotted_back_face": CLEAR_SLOTTED_BACK_FACE_CLASS,
        "side_port": SIDE_PORT_FACE_CLASS,
        "ambiguous": AMBIGUOUS_FACE_CLASS,
        "button_front_face": BUTTON_FACE_CLASS,
    }
    resolved_face_class = face_classes[face_class]
    visible = resolved_face_class == BUTTON_FACE_CLASS
    resolved_camera = gate_camera if gate_camera == "head" else f"{press_hand}_wrist"
    frame_id = f"{resolved_camera}:17:frame"
    facade._prepress_gate = {
        "button_visible": visible,
        "face_class": resolved_face_class,
        "negative_case": None if visible else face_class,
        "gate_id": "button_gate_test",
        "env_step": 17,
        "frame_id": frame_id,
        "capture_group_id": "capture:17",
        "camera": gate_camera,
        "resolved_camera": resolved_camera,
    }
    facade._prepress_projection = (
        {
            "projection_id": "button_projection_test",
            "gate_id": "button_gate_test",
            "camera": gate_camera,
            "resolved_camera": resolved_camera,
            "frame_id": frame_id,
            "capture_group_id": "capture:17",
            "env_step": 17,
            "button_center_world": [0.05292, 0.03502, 0.58661],
            "button_normal_world": [0.7740, -0.6332, 0.0],
        }
        if visible
        else None
    )
    frame = SimpleNamespace(
        step_index=17,
        capture_group_id="capture:17",
        frame_id=frame_id,
        camera_to_world=np.eye(4),
    )
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, requested_frame_id: frame,
        latest=lambda camera: frame,
    )
    stability = {
        "stable": True,
        "held_eef_position_world": [0.0, 0.0, 0.60],
        "held_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "press_eef_position_world": [0.0, 0.0, 0.60],
        "press_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "radio_position_world": [0.0, 0.0, 0.60],
        "radio_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "relative_position_m": [0.0, 0.0, 0.0],
        "relative_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "air_gap_m": 0.20,
    }
    facade._assert_prepress_checkpoint_bound = lambda: (None, {})
    facade._prepress_stability_snapshot = lambda: dict(stability)
    facade._require_planner = lambda: SimpleNamespace(backend=Backend())
    facade._resolve_handoff_targets = lambda: (
        SimpleNamespace(root_link=root),
        object(),
    )
    facade._robot = lambda: Robot()
    facade._object_pose = lambda _obj: (
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    goal_move_to = facade.prepress_move_to

    def dispatch_test_move_to(**kwargs):
        # Existing tests below exercise the private resolved-candidate safety
        # layer. The real public method is separately tested with button_goal
        # and never accepts these literal pose fields.
        if "target_xyz" in kwargs or "target_quat_xyzw" in kwargs:
            return facade._prepress_move_to_candidate(**kwargs)
        return goal_move_to(**kwargs)

    facade.prepress_move_to = dispatch_test_move_to
    return facade, calls


def _assert_motion_rejected_before_planner(facade, calls, **kwargs) -> None:
    """Accept either public failure or an exception, but never a planner call."""

    try:
        result = facade.prepress_move_to(plan_only=True, **kwargs)
    except (RuntimeError, ValueError):
        result = None
    if result is not None:
        assert result["primitive_success"] is False
    assert not any(call[0] == "plan" for call in calls)


def test_button_gate_requires_complete_positive_signature_before_coordinates():
    result = validate_button_declaration(
        button_visible=True,
        positive_signature={**_signature(), "white_outer_ring": False},
        negative_case=None,
        bbox_xyxy=None,
        center_uv=None,
        image_width=640,
        image_height=480,
    )
    assert result["verdict"] == "NOT_VISIBLE"
    assert result["bbox_xyxy"] is None
    assert result["center_uv"] is None

    with pytest.raises(ValueError, match="failed button signature"):
        validate_button_declaration(
            button_visible=True,
            positive_signature={**_signature(), "red_center_bump": False},
            negative_case=None,
            bbox_xyxy=[10, 10, 30, 30],
            center_uv=[20, 20],
            image_width=640,
            image_height=480,
        )


def test_not_visible_rejects_coordinates_and_visible_accepts_bounded_center():
    with pytest.raises(ValueError, match="NOT_VISIBLE"):
        validate_button_declaration(
            button_visible=False,
            positive_signature=None,
            negative_case="side_port",
            bbox_xyxy=None,
            center_uv=[20, 20],
            image_width=640,
            image_height=480,
        )
    visible = validate_button_declaration(
        button_visible=True,
        positive_signature=_signature(),
        negative_case=None,
        bbox_xyxy=[10, 12, 30, 34],
        center_uv=[20, 23],
        image_width=640,
        image_height=480,
    )
    assert visible["verdict"] == "VISIBLE"
    assert visible["positive_signature_complete"] is True

    with pytest.raises(ValueError, match="failed button signature"):
        validate_button_declaration(
            button_visible=True,
            positive_signature=(
                "not red front; no black round disk; no white ring; no red center bump"
            ),
            negative_case=None,
            bbox_xyxy=[10, 12, 30, 34],
            center_uv=[20, 23],
            image_width=640,
            image_height=480,
        )


def test_upright_radio_orientation_resolves_exact_opposite_face_with_two_axes():
    target_front = np.asarray([-1.0, 0.0, 0.0])
    quaternion = upright_radio_orientation_xyzw(target_front)
    local_front = np.asarray([0.7740, -0.6332, 0.0])
    local_front /= np.linalg.norm(local_front)

    np.testing.assert_allclose(
        quat_rotate_xyzw(quaternion, local_front),
        target_front,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        quat_rotate_xyzw(quaternion, [0.0, 0.0, 1.0]),
        [0.0, 0.0, 1.0],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "face_class",
    ["clear_slotted_back_face", "side_port", "ambiguous"],
)
def test_button_validator_emits_canonical_negative_face_class(face_class):
    result = validate_button_declaration(
        button_visible=False,
        positive_signature=None,
        negative_case=face_class,
        bbox_xyxy=None,
        center_uv=None,
        image_width=640,
        image_height=480,
    )

    assert result["button_visible"] is False
    assert result["verdict"] == "NOT_VISIBLE"
    assert (
        result["face_class"]
        == {
            "clear_slotted_back_face": CLEAR_SLOTTED_BACK_FACE_CLASS,
            "side_port": SIDE_PORT_FACE_CLASS,
            "ambiguous": AMBIGUOUS_FACE_CLASS,
        }[face_class]
    )
    assert result["negative_case"] == face_class
    assert result["bbox_xyxy"] is None
    assert result["center_uv"] is None


@pytest.mark.parametrize("negative_case", ["red_side_slot", "side_slot", "other"])
def test_button_validator_rejects_noncanonical_negative_case(negative_case):
    with pytest.raises(ValueError, match="negative_case"):
        validate_button_declaration(
            button_visible=False,
            positive_signature=None,
            negative_case=negative_case,
            bbox_xyxy=None,
            center_uv=None,
            image_width=640,
            image_height=480,
        )


def test_complete_button_signature_retains_button_front_face_class():
    visible = validate_button_declaration(
        button_visible=True,
        positive_signature=_signature(),
        negative_case=None,
        bbox_xyxy=[10, 12, 30, 34],
        center_uv=[20, 23],
        image_width=640,
        image_height=480,
    )

    assert visible["verdict"] == "VISIBLE"
    assert visible["face_class"] == BUTTON_FACE_CLASS
    assert visible["positive_signature_complete"] is True
    assert visible["negative_case"] is None


def test_geometry_uses_press_line_opposition_and_axial_standoff_gates():
    aligned = evaluate_geometry(
        button_center_world=[0.0, 0.0, 0.045],
        button_normal_world=[0.0, 0.0, -1.0],
        press_eef_position_world=[0.0, 0.0, 0.0],
        press_direction_world=[0.0, 0.0, 1.0],
    )
    assert aligned["geometry_pass"] is True
    assert aligned["button_center_to_press_approach_line_m"] == pytest.approx(0.0)
    assert aligned["button_normal_opposition_angle_deg"] == pytest.approx(0.0)
    assert aligned["axial_standoff_m"] == pytest.approx(0.045)

    offset = evaluate_geometry(
        button_center_world=[0.011, 0.0, 0.045],
        button_normal_world=[0.0, 0.0, -1.0],
        press_eef_position_world=[0.0, 0.0, 0.0],
        press_direction_world=[0.0, 0.0, 1.0],
    )
    assert offset["geometry_pass"] is False
    assert offset["criteria"]["line_distance"] is False


def test_press_direction_is_eef_local_positive_z():
    identity = quat_rotate_xyzw([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert np.allclose(identity, [0.0, 0.0, 1.0])

    delta = quat_between_vectors_xyzw([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(
        quat_rotate_xyzw(delta, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
    )


def test_prepress_public_env_methods_and_specialized_generator_exist():
    expected = {
        "inspect_post_pick_state",
        "declare_button_visibility",
        "project_button",
        "evaluate_prepress_geometry",
        "prepress_move_to",
        "prepress_rotate_wrist",
        "save_prepress_checkpoint",
    }
    assert all(callable(getattr(BehaviorEnvFacade, name, None)) for name in expected)
    public_source = inspect.getsource(BehaviorEnvFacade.prepress_move_to)
    motion_source = inspect.getsource(BehaviorEnvFacade._prepress_move_to_candidate)
    assert "generate_button_goal_pose_candidates" in public_source
    assert "generate_press_staging_pose_candidates" in public_source
    assert (
        "target_xyz"
        not in inspect.signature(BehaviorEnvFacade.prepress_move_to).parameters
    )
    assert 'kind="prepress_arm"' in motion_source
    assert "hand=active" in motion_source
    assert "arm_idx.get(locked" in motion_source
    assert 'active = held if role == "held" else press' in motion_source
    assert "EEF_LINK_BY_HAND[held]" in motion_source
    assert "reported_keys != {expected_attachment_link}" in motion_source
    assert "len(trunk_lock_indices) != 4" in motion_source
    assert "len(inactive_arm_indices) != 7" in motion_source
    assert "held}_gripper" in motion_source
    waypoint_loop = motion_source[
        motion_source.index("for index, waypoint") : motion_source.index(
            "self._finalize_video_segment()"
        )
    ]
    assert "_prepress_stability_snapshot" not in waypoint_loop
    assert "held_gripper_command" in waypoint_loop
    assert "press_contact_count" in inspect.getsource(
        BehaviorEnvFacade._prepress_stability_snapshot
    )
    assert callable(getattr(RealCuroboBackend, "plan_prepress_arm_trajectory"))


def test_post_pick_pixel_to_world_records_projection_only_for_current_gate_center():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 42
    facade._prepress_projection = None
    facade._prepress_geometry = {"stale": True}
    facade._prepress_gate = {
        "button_visible": True,
        "face_class": BUTTON_FACE_CLASS,
        "gate_id": "gate-42",
        "camera": "head",
        "resolved_camera": "head",
        "frame_id": "head:42:abc",
        "capture_group_id": "capture:42:abc",
        "center_uv": [20.0, 30.0],
        "env_step": 42,
    }
    facade._pixel_to_world_raw = lambda **_kwargs: {
        "_finish": False,
        "primitive_success": True,
        "stop_reason": "pixel_projected",
        "metrics": {"confidence": 0.99},
        "diagnostics": {
            "xyz": [1.0, 2.0, 3.0],
            "surface_normal": [0.0, 0.0, 1.0],
        },
        "total_env_steps": 42,
    }

    safe_pixel = facade.pixel_to_world(
        camera="head",
        frame_id="head:42:abc",
        u=10,
        v=12,
    )
    assert safe_pixel["stop_reason"] == "pixel_projected"
    assert facade._prepress_projection is None

    button = facade.pixel_to_world(
        camera="head",
        frame_id="head:42:abc",
        u=20,
        v=30,
    )
    assert button["stop_reason"] == "button_projected"
    assert button["button_center_world"] == [1.0, 2.0, 3.0]
    assert button["button_normal_world"] == [0.0, 0.0, 1.0]
    assert button["projection_id"].startswith("button_projection_")
    assert facade._prepress_projection["env_step"] == 42
    assert facade._prepress_geometry is None


def test_prepress_rotate_wrist_plan_only_binds_dynamic_role_without_hand():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._assert_prepress_checkpoint_bound = lambda: (None, {})
    facade._prepress_stability_snapshot = lambda: {
        "held_eef_position_world": [1.0, 2.0, 0.7],
        "held_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "press_eef_position_world": [3.0, 4.0, 0.8],
        "press_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    calls = []
    facade._prepress_move_to_candidate = lambda **kwargs: (
        calls.append(kwargs)
        or {
            "primitive_success": True,
            "plan_only": kwargs["plan_only"],
        }
    )

    result = facade.prepress_rotate_wrist(
        role="press",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.02],
        frame="eef",
        plan_only=True,
        timeout_s=12.0,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    assert calls[0]["role"] == "press"
    assert calls[0]["target_xyz"] == [3.0, 4.0, 0.8]
    assert calls[0]["plan_only"] is True
    assert calls[0]["timeout_s"] == 12.0
    assert calls[0]["press_observation_rotation"] is True
    assert np.allclose(
        calls[0]["target_quat_xyzw"],
        [0.0, 0.0, np.sin(0.01), np.cos(0.01)],
    )


def test_press_observation_rotation_authorizes_only_zero_translation_without_gate():
    half_turn = [0.0, 0.0, 1.0, 0.0]
    allowed = authorize_prepress_motion(
        role="press",
        current_xyz=[1.0, 2.0, 3.0],
        current_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_xyz=[1.0, 2.0, 3.0],
        target_quat_xyzw=half_turn,
        face_class=None,
        press_observation_rotation=True,
    )

    assert allowed["allowed"] is True
    assert allowed["policy"] == "press_in_place_observation_rotation"
    assert allowed["translation_limit_m"] == 0.0
    assert allowed["rotation_limit_rad"] == pytest.approx(np.pi)
    assert allowed["rotation_rad"] == pytest.approx(np.pi)

    translated = authorize_prepress_motion(
        role="press",
        current_xyz=[1.0, 2.0, 3.0],
        current_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_xyz=[1.000001, 2.0, 3.0],
        target_quat_xyzw=half_turn,
        face_class=None,
        press_observation_rotation=True,
    )
    assert translated["allowed"] is False


def test_press_rotate_wrist_without_projection_reaches_certified_curobo_plan(
    tmp_path,
):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    facade._prepress_projection = None

    result = facade.prepress_rotate_wrist(
        role="press",
        relative_axis_angle=[0.0, 1.0, 0.0, np.pi],
        frame="world",
        plan_only=True,
        timeout_s=12.0,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    assert result["motion_authorization"]["policy"] == (
        "press_in_place_observation_rotation"
    )
    assert result["plan_certificate"]["gate_binding"] is None
    assert result["locked_hand"] == "right"
    planned = next(call[1] for call in calls if call[0] == "plan")
    assert planned["hand"] == "left"
    assert planned["timeout_s"] == 12.0
    collision = next(call for call in calls if call[0] == "collision")
    assert set(collision[2]) == {EEF_LINK_BY_HAND["right"]}


def test_press_observation_rotation_rejects_any_translation_before_planner(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    facade._prepress_projection = None

    result = facade._prepress_move_to_candidate(
        role="press",
        target_xyz=[0.000001, 0.0, 0.60],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        plan_only=True,
        press_observation_rotation=True,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "search_motion_too_large_for_visual_gate"
    assert result["motion_authorization"]["policy"] == (
        "press_in_place_observation_rotation"
    )
    assert not any(call[0] == "plan" for call in calls)


def test_press_observation_rotation_still_requires_plan_only_certificate(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    facade._prepress_projection = None

    result = facade.prepress_rotate_wrist(
        role="press",
        relative_axis_angle=[1.0, 0.0, 0.0, 0.2],
        frame="eef",
        plan_only=False,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "matching_plan_only_certificate_required"
    assert result["motion_authorization"]["policy"] == (
        "press_in_place_observation_rotation"
    )
    assert not any(call[0] == "plan" for call in calls)


def test_press_observation_rotation_rejects_relative_angle_over_pi(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    facade._prepress_projection = None

    with pytest.raises(ValueError, match="must not exceed pi"):
        facade.prepress_rotate_wrist(
            role="press",
            relative_axis_angle=[1.0, 0.0, 0.0, np.pi + 1e-6],
            frame="eef",
            plan_only=True,
        )

    assert not any(call[0] == "plan" for call in calls)


@pytest.mark.parametrize(
    ("held", "press"),
    [("right", "left"), ("left", "right")],
)
def test_post_pick_camera_roles_resolve_from_dynamic_hands(held, press):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._prepress_context = {
        "held_hand": held,
        "press_hand": press,
    }

    assert facade._resolve_post_pick_camera("head") == "head"
    assert facade._resolve_post_pick_camera("held_wrist") == f"{held}_wrist"
    assert facade._resolve_post_pick_camera("press_wrist") == f"{press}_wrist"

    facade._prepress_context = None
    with pytest.raises(RuntimeError, match="inspect_post_pick_state"):
        facade._resolve_post_pick_camera("held_wrist")


def test_post_pick_camera_role_round_trip_stays_public_while_using_physical_camera():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    facade._env_steps = 9
    facade._prepress_context = {
        "held_hand": "right",
        "press_hand": "left",
    }
    facade._prepress_motion = None
    facade._prepress_gate = None
    frame = SimpleNamespace(
        frame_id="left:9:frame",
        timestamp_s=time.monotonic(),
        step_index=9,
        capture_group_id="capture:9",
    )

    class _FrameCache:
        def latest(self, camera):
            assert camera == "left_wrist"
            return frame

        def get_current(self, camera, frame_id):
            assert camera == "left_wrist"
            assert frame_id == frame.frame_id
            return frame

    facade._frame_cache = _FrameCache()
    facade._require_planner = lambda: SimpleNamespace(
        observe=lambda camera: {
            "camera": camera,
            "frame_id": frame.frame_id,
        }
    )
    facade._persist_live_observation = lambda payload: dict(payload)
    observed = facade.observe("press_wrist")
    assert observed["camera"] == "press_wrist"
    assert observed["resolved_camera"] == "left_wrist"

    facade._pixel_to_world_raw = lambda **kwargs: {
        "primitive_success": True,
        "stop_reason": "projected",
        "diagnostics": {"xyz": [1.0, 2.0, 3.0]},
    }
    projected = facade.pixel_to_world(
        camera=observed["camera"],
        frame_id=observed["frame_id"],
        u=10,
        v=12,
    )
    assert projected["camera"] == "press_wrist"
    assert projected["resolved_camera"] == "left_wrist"


def test_inspect_post_pick_state_returns_dynamic_transform_context(tmp_path):
    paused_runtime_path = tmp_path / "paused_runtime.json"
    paused_runtime_path.write_text(
        json.dumps(
            {
                "handoff_state": "PAUSED",
                "action_source": "curobo",
                "vla_actions_enabled": False,
                "vla_action_gate_confirmed": True,
                "lifecycle_finalized": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "state_checkpoint_1.json"
    checkpoint = {
        "held_hand": "right",
        "press_hand": "left",
        "object_name": "radio_89",
    }
    stability = {
        "stable": True,
        "held_eef_position_world": [1.0, 2.0, 0.70],
        "held_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "press_eef_position_world": [1.1, 2.2, 0.75],
        "press_eef_quat_xyzw": [0.0, 0.0, 1.0, 0.0],
        "radio_position_world": [0.9, 2.0, 0.65],
        "radio_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "relative_position_m": [-0.1, 0.0, -0.05],
        "relative_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._handoff_state = "PAUSED"
    facade._action_source = "curobo"
    facade._vla_actions_enabled = False
    facade._paused_runtime_path = paused_runtime_path
    facade._env_steps = 23
    facade._last_info = {"done": {"success": False}}
    facade._read_post_pick_checkpoint = lambda: (
        checkpoint_path,
        dict(checkpoint),
        "checkpoint-sha256",
    )
    facade._refresh_observation_without_step = lambda: None
    facade._prepress_stability_snapshot = lambda: dict(stability)

    result = facade.inspect_post_pick_state()

    transform = result["transform_context"]
    assert "T_A_B maps coordinates" in transform["convention"]
    assert np.asarray(transform["T_world_held_current"]).shape == (4, 4)
    assert np.asarray(transform["T_world_press_current"]).shape == (4, 4)
    assert np.asarray(transform["T_world_radio_current"]).shape == (4, 4)
    assert np.asarray(transform["T_held_radio_current"]).shape == (4, 4)
    np.testing.assert_allclose(
        np.asarray(transform["T_world_held_current"])[:3, 3],
        stability["held_eef_position_world"],
    )
    np.testing.assert_allclose(
        np.asarray(transform["T_world_press_current"])[:3, 3],
        stability["press_eef_position_world"],
    )
    np.testing.assert_allclose(
        np.asarray(transform["T_world_radio_current"])[:3, 3],
        stability["radio_position_world"],
    )
    assert (
        "runtime applies T_held_radio_current internally"
        in transform["candidate_generation"]
    )
    assert (
        "must not submit a literal held EEF pose" in transform["candidate_generation"]
    )
    assert transform["radio_pose_prior"]["local_button_center_m"]
    assert transform["radio_pose_prior"]["local_button_face_outward_normal"]


def test_button_goal_candidates_target_button_state_not_literal_eef_pose():
    world_held = np.eye(4)
    world_held[2, 3] = 0.60
    world_radio = world_held.copy()
    world_press = np.eye(4)
    world_press[:3, 3] = [1.0, 0.0, 0.60]
    result = generate_button_goal_pose_candidates(
        world_held_transform=world_held,
        world_radio_transform=world_radio,
        held_to_radio_transform=np.eye(4),
        button_center_world=[0.05, 0.0, 0.60],
        button_normal_world=[1.0, 0.0, 0.0],
        world_press_transform=world_press,
        goal={
            "chest_direction_world": [-1.0, 0.0, 0.0],
            "chest_translation_m": 0.10,
            "position_perturbations_world_m": [[0.0, 0.0, 0.0]],
            "orientation_perturbations_world_axis_angle": [[0.0, 0.0, 1.0, 0.0]],
            "normal_blend_factors": [1.0],
            "max_position_perturbation_m": 0.0,
            "max_orientation_perturbation_rad": 0.0,
            "max_press_approach_opposition_angle_deg": 180.0,
            "max_candidates": 1,
        },
        head_optical_axis_world=[0.0, 0.0, -1.0],
    )
    candidate = result["candidates"][0]
    assert candidate["eligible"] is True
    np.testing.assert_allclose(
        candidate["predicted_button_center_world"], [-0.05, 0.0, 0.60]
    )
    assert candidate["geometry"]["head_side_view_error_deg"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        candidate["target_held_eef_pose"]["matrix"],
        candidate["desired_radio_pose"]["matrix"],
    )


def test_press_staging_candidates_are_projection_geometry_driven():
    candidates = generate_press_staging_pose_candidates(
        button_center_world=[0.4, 0.1, 0.8],
        button_normal_world=[1.0, 0.0, 0.0],
        world_press_transform=np.eye(4),
        standoff_m=0.055,
        max_candidates=4,
    )["candidates"]
    assert len(candidates) == 4
    assert all(candidate["eligible"] for candidate in candidates)
    for candidate in candidates:
        np.testing.assert_allclose(
            candidate["target_press_eef_pose"]["position"], [0.455, 0.1, 0.8]
        )
        assert candidate["geometry"]["button_normal_opposition_angle_deg"] < 1e-6


def test_far_press_observation_staging_is_eligible_but_not_final_prepress():
    eef_to_camera = np.eye(4)
    eef_to_camera[:3, 3] = [-0.05, 0.003, -0.065]
    candidates = generate_press_staging_pose_candidates(
        button_center_world=[0.4, 0.1, 0.8],
        button_normal_world=[1.0, 0.0, 0.0],
        world_press_transform=np.eye(4),
        standoff_m=0.15,
        max_candidates=4,
        alignment_phase="observation",
        eef_to_camera_transform=eef_to_camera,
    )["candidates"]
    assert len(candidates) == 4
    assert all(candidate["eligible"] for candidate in candidates)
    assert not any(
        candidate["final_prepress_geometry_pass"] for candidate in candidates
    )
    for candidate in candidates:
        target_eef = pose_matrix_xyzw(
            candidate["target_press_eef_pose"]["position"],
            candidate["target_press_eef_pose"]["quat_xyzw"],
        )
        target_camera = target_eef @ eef_to_camera
        point_camera = np.linalg.inv(target_camera) @ [0.4, 0.1, 0.8, 1.0]
        np.testing.assert_allclose(point_camera[:2], [0.0, 0.0], atol=1e-8)
        assert point_camera[2] == pytest.approx(-0.15)
        assert candidate["predicted_button_camera_center_angle_deg"] < 1e-6


def test_button_goal_search_skips_unreachable_candidate_and_certifies_next(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="clear_slotted_back_face")
    backend = facade._require_planner().backend
    original_plan = backend.plan_prepress_arm_trajectory
    plan_count = 0

    def fail_first(**kwargs):
        nonlocal plan_count
        plan_count += 1
        if plan_count == 1:
            calls.append(("plan", kwargs))
            return {"ok": False, "stop_reason": "unreachable", "metrics": {}}
        return original_plan(**kwargs)

    backend.plan_prepress_arm_trajectory = fail_first
    facade._require_planner = lambda: SimpleNamespace(backend=backend)
    result = facade.prepress_move_to(
        role="held", button_goal=_held_button_goal(), plan_only=True
    )
    assert result["primitive_success"] is True
    assert len(result["candidate_attempts"]) == 2
    assert result["candidate_attempts"][0]["stop_reason"] == "unreachable"
    assert result["plan_certificate"]["goal_binding"]["role"] == "held"


def test_visible_button_head_pixel_goal_drives_button_space_translation(tmp_path):
    facade, _calls = _motion_guard_facade(tmp_path, face_class="button_front_face")
    frame = facade._frame_cache.latest("head")
    frame.intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
    )
    frame.correction_profile = None
    facade._prepress_projection["button_center_world"] = [0.05, 0.03, -1.0]
    stability = facade._prepress_stability_snapshot()
    stability.update(
        {
            "held_eef_position_world": [0.0, 0.0, -1.0],
            "press_eef_position_world": [0.0, -0.5, -1.0],
            "radio_position_world": [0.0, 0.0, -1.0],
        }
    )
    facade._prepress_stability_snapshot = lambda: dict(stability)
    goal = _held_button_goal(0.0)
    goal["head_target_uv"] = [50.0, 75.0]
    goal["head_target_radius_px"] = 0.0
    goal["alignment_phase"] = "position_first"

    result = facade.prepress_move_to(role="held", button_goal=goal, plan_only=True)

    assert result["primitive_success"] is True
    selected = result["selected_candidate"]
    assert selected["translation_source"] == "head_button_center_pixel"
    assert selected["alignment_phase"] == "position_first"
    assert selected["head_target_uv"] == [50.0, 75.0]
    np.testing.assert_allclose(
        selected["desired_radio_pose"]["quat_xyzw"],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        selected["predicted_button_center_world"],
        [0.0, -0.25, -1.0],
        atol=1e-6,
    )


def test_head_pixel_goal_requires_direct_visible_button(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="clear_slotted_back_face")
    goal = _held_button_goal(0.0)
    goal["head_target_uv"] = [50.0, 75.0]
    goal["alignment_phase"] = "position_first"

    with pytest.raises(RuntimeError, match="directly visible button projection"):
        facade.prepress_move_to(role="held", button_goal=goal, plan_only=True)
    assert not any(call[0] == "plan" for call in calls)


def test_normal_refine_uses_fresh_head_and_radio_local_button_prior(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="button_front_face")
    facade._prepress_gate = None
    facade._prepress_projection = None
    goal = _held_button_goal(0.0)
    goal["alignment_phase"] = "normal_refine"
    goal["position_slack_m"] = 0.02

    result = facade.prepress_move_to(role="held", button_goal=goal, plan_only=True)

    assert result["primitive_success"] is True
    assert (
        result["goal_binding"]["geometry_source"]
        == "radio_local_button_prior_normal_refine"
    )
    assert result["goal_binding"]["gate_id"]
    assert any(call[0] == "plan" for call in calls)


def test_button_goal_all_candidates_fail_without_advancing_physics(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="clear_slotted_back_face")
    backend = facade._require_planner().backend

    def always_fail(**kwargs):
        calls.append(("plan", kwargs))
        return {"ok": False, "stop_reason": "unreachable", "metrics": {}}

    backend.plan_prepress_arm_trajectory = always_fail
    facade._require_planner = lambda: SimpleNamespace(backend=backend)
    start_step = facade._env_steps
    result = facade.prepress_move_to(
        role="held", button_goal=_held_button_goal(), plan_only=True
    )
    assert result["stop_reason"] == "button_goal_unreachable"
    assert result["candidate_attempts"]
    assert facade._env_steps == start_step


@pytest.mark.parametrize(
    ("held_hand", "press_hand"), [("right", "left"), ("left", "right")]
)
def test_button_goal_roles_resolve_dynamically(tmp_path, held_hand, press_hand):
    held_facade, held_calls = _motion_guard_facade(
        tmp_path / "held",
        face_class="clear_slotted_back_face",
        held_hand=held_hand,
        press_hand=press_hand,
    )
    held = held_facade.prepress_move_to(
        role="held", button_goal=_held_button_goal(), plan_only=True
    )
    assert held["primitive_success"] is True
    assert (
        next(call[1]["hand"] for call in held_calls if call[0] == "plan") == held_hand
    )

    press_facade, press_calls = _motion_guard_facade(
        tmp_path / "press",
        face_class="button_front_face",
        held_hand=held_hand,
        press_hand=press_hand,
        gate_camera="press_wrist",
    )
    press = press_facade.prepress_move_to(
        role="press", button_goal=_press_button_goal(), plan_only=True
    )
    assert press["primitive_success"] is True
    assert (
        next(call[1]["hand"] for call in press_calls if call[0] == "plan") == press_hand
    )


def test_button_goals_fail_closed_on_wrong_visual_provenance(tmp_path):
    held, held_calls = _motion_guard_facade(tmp_path / "held", face_class="ambiguous")
    with pytest.raises(RuntimeError, match="BUTTON_FACE or clear slotted"):
        held.prepress_move_to(
            role="held", button_goal=_held_button_goal(), plan_only=True
        )
    assert not any(call[0] == "plan" for call in held_calls)

    press, press_calls = _motion_guard_facade(
        tmp_path / "press", face_class="button_front_face", gate_camera="head"
    )
    with pytest.raises(RuntimeError, match="dynamic press-wrist"):
        press.prepress_move_to(
            role="press", button_goal=_press_button_goal(), plan_only=True
        )
    assert not any(call[0] == "plan" for call in press_calls)


def test_button_goal_execution_rejects_changed_goal_after_plan(tmp_path):
    facade, _calls = _motion_guard_facade(
        tmp_path, face_class="clear_slotted_back_face"
    )
    planned = facade.prepress_move_to(
        role="held", button_goal=_held_button_goal(0.10), plan_only=True
    )
    assert planned["primitive_success"] is True
    changed = facade.prepress_move_to(
        role="held", button_goal=_held_button_goal(0.11), plan_only=False
    )
    assert changed["stop_reason"] == "button_goal_changed_after_plan_only"


@pytest.mark.parametrize("face_class", ["side_port", "ambiguous"])
@pytest.mark.parametrize(
    ("target_xyz", "target_quat_xyzw"),
    [
        ([0.050001, 0.0, 0.60], [0.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.60], _z_quat(15.001)),
        ([0.051, 0.0, 0.60], _z_quat(16.0)),
    ],
)
def test_uncertain_face_rejects_large_move_before_planner(
    tmp_path,
    face_class,
    target_xyz,
    target_quat_xyzw,
):
    facade, calls = _motion_guard_facade(tmp_path, face_class=face_class)

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="held",
        target_xyz=target_xyz,
        target_quat_xyzw=target_quat_xyzw,
    )


@pytest.mark.parametrize("face_class", ["side_port", "ambiguous"])
@pytest.mark.parametrize(
    ("target_xyz", "target_quat_xyzw"),
    [
        ([0.05, 0.0, 0.60], [0.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.60], _z_quat(15.0)),
    ],
)
def test_uncertain_face_allows_motion_at_declared_guard_boundary(
    tmp_path,
    face_class,
    target_xyz,
    target_quat_xyzw,
):
    facade, calls = _motion_guard_facade(tmp_path, face_class=face_class)

    result = facade.prepress_move_to(
        role="held",
        target_xyz=target_xyz,
        target_quat_xyzw=target_quat_xyzw,
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    assert any(call[0] == "plan" for call in calls)


def test_clear_slotted_back_face_allows_one_direct_held_move(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="clear_slotted_back_face",
    )

    result = facade.prepress_move_to(
        role="held",
        target_xyz=[0.60, 0.0, 0.60],
        target_quat_xyzw=upright_radio_orientation_xyzw([-1.0, 0.0, 0.0]),
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    planned = next(call[1] for call in calls if call[0] == "plan")
    assert planned["hand"] == "right"


def test_clear_slotted_back_face_coarse_flip_is_stage_one_shot(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="clear_slotted_back_face",
    )
    facade._prepress_coarse_flip_used = True

    result = facade.prepress_move_to(
        role="held",
        target_xyz=[0.60, 0.0, 0.60],
        target_quat_xyzw=upright_radio_orientation_xyzw([-1.0, 0.0, 0.0]),
        plan_only=True,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "coarse_back_to_front_flip_already_used"
    assert not any(call[0] == "plan" for call in calls)


@pytest.mark.parametrize(
    ("held_hand", "press_hand"),
    [("right", "left"), ("left", "right")],
)
def test_prepress_motion_roles_are_dynamic_for_both_hands(
    tmp_path,
    held_hand,
    press_hand,
):
    held_facade, held_calls = _motion_guard_facade(
        tmp_path / "held",
        face_class="clear_slotted_back_face",
        held_hand=held_hand,
        press_hand=press_hand,
    )
    held_result = held_facade.prepress_move_to(
        role="held",
        target_xyz=[0.60, 0.0, 0.60],
        target_quat_xyzw=upright_radio_orientation_xyzw([-1.0, 0.0, 0.0]),
        plan_only=True,
    )
    assert held_result["primitive_success"] is True
    assert (
        next(call[1]["hand"] for call in held_calls if call[0] == "plan") == held_hand
    )

    press_facade, press_calls = _motion_guard_facade(
        tmp_path / "press",
        face_class="button_front_face",
        held_hand=held_hand,
        press_hand=press_hand,
    )
    press_result = press_facade.prepress_move_to(
        role="press",
        target_xyz=[0.30, 0.0, 0.60],
        target_quat_xyzw=_z_quat(120.0),
        plan_only=True,
    )
    assert press_result["primitive_success"] is True
    assert (
        next(call[1]["hand"] for call in press_calls if call[0] == "plan") == press_hand
    )


def test_clear_slotted_back_face_rejects_press_before_positive_projection(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="clear_slotted_back_face",
    )

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="press",
        target_xyz=[0.30, 0.0, 0.60],
        target_quat_xyzw=_z_quat(120.0),
    )


def test_clear_slotted_back_face_rejects_translation_over_direct_limit(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="clear_slotted_back_face",
    )

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="held",
        target_xyz=[0.600001, 0.0, 0.60],
        target_quat_xyzw=upright_radio_orientation_xyzw([-1.0, 0.0, 0.0]),
    )


def test_clear_slotted_back_face_rejects_arbitrary_non_upright_flip(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="clear_slotted_back_face",
    )

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="held",
        target_xyz=[0.30, 0.0, 0.60],
        target_quat_xyzw=[1.0, 0.0, 0.0, 0.0],
    )


def test_execution_requires_exact_one_use_plan_only_certificate(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    target = {
        "role": "held",
        "target_xyz": [0.01, 0.0, 0.60],
        "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    without_plan = facade.prepress_move_to(plan_only=False, **target)
    assert without_plan["primitive_success"] is False
    assert without_plan["stop_reason"] == "matching_plan_only_certificate_required"
    assert not any(call[0] == "plan" for call in calls)

    planned = facade.prepress_move_to(plan_only=True, **target)
    assert planned["primitive_success"] is True
    assert planned["plan_certificate"]["role"] == "held"
    planner_calls = len([call for call in calls if call[0] == "plan"])

    changed = facade.prepress_move_to(
        role="held",
        target_xyz=[0.011, 0.0, 0.60],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        plan_only=False,
    )
    assert changed["primitive_success"] is False
    assert changed["stop_reason"] == "matching_plan_only_certificate_required"
    assert len([call for call in calls if call[0] == "plan"]) == planner_calls

    consumed = facade.prepress_move_to(plan_only=False, **target)
    assert consumed["primitive_success"] is False
    assert consumed["stop_reason"] == "matching_plan_only_certificate_required"


def test_execution_consumes_cached_trajectory_and_rejects_changed_start_state(
    tmp_path,
):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")
    target = {
        "role": "held",
        "target_xyz": [0.01, 0.0, 0.60],
        "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    planned = facade.prepress_move_to(plan_only=True, **target)
    assert planned["primitive_success"] is True
    planner_calls = len([call for call in calls if call[0] == "plan"])
    facade._robot = lambda: SimpleNamespace(
        get_joint_positions=lambda: np.full(24, 0.01),
        base_idx=np.arange(6),
        trunk_control_idx=np.arange(6, 10),
        arm_control_idx={
            "left": np.arange(10, 17),
            "right": np.arange(17, 24),
        },
    )

    with pytest.raises(RuntimeError, match="start state changed"):
        facade.prepress_move_to(plan_only=False, **target)
    assert len([call for call in calls if call[0] == "plan"]) == planner_calls


def test_button_front_face_allows_llm_selected_held_micro_adjustment(tmp_path):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="button_front_face",
    )

    result = facade.prepress_move_to(
        role="held",
        target_xyz=[0.08, 0.0, 0.60],
        target_quat_xyzw=_z_quat(30.0),
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    planned = next(call[1] for call in calls if call[0] == "plan")
    assert planned["hand"] == "right"


@pytest.mark.parametrize(
    ("target_xyz", "target_quat_xyzw"),
    [
        ([0.080001, 0.0, 0.60], [0.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.60], _z_quat(30.001)),
    ],
)
def test_button_front_face_rejects_motion_over_micro_adjustment_limit(
    tmp_path,
    target_xyz,
    target_quat_xyzw,
):
    facade, calls = _motion_guard_facade(
        tmp_path,
        face_class="button_front_face",
    )

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="held",
        target_xyz=target_xyz,
        target_quat_xyzw=target_quat_xyzw,
    )


@pytest.mark.parametrize("face_class", ["side_port", "ambiguous"])
def test_safe_vertical_lift_does_not_authorize_press_before_projection(
    tmp_path,
    face_class,
):
    held_facade, held_calls = _motion_guard_facade(
        tmp_path / "held",
        face_class=face_class,
    )
    held = held_facade.prepress_move_to(
        role="held",
        target_xyz=[0.0, 0.0, 0.70],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        plan_only=True,
    )
    assert held["primitive_success"] is True
    assert any(call[0] == "plan" for call in held_calls)

    press_facade, press_calls = _motion_guard_facade(
        tmp_path / "press",
        face_class=face_class,
    )
    _assert_motion_rejected_before_planner(
        press_facade,
        press_calls,
        role="press",
        target_xyz=[0.0, 0.0, 0.70],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )


def test_projected_press_staging_does_not_inherit_held_micro_limits(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="button_front_face")

    result = facade.prepress_move_to(
        role="press",
        target_xyz=[0.30, 0.0, 0.60],
        target_quat_xyzw=_z_quat(120.0),
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert result["plan_only"] is True
    assert any(call[0] == "plan" for call in calls)


def test_press_staging_rejects_positive_gate_without_fresh_projection(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="button_front_face")
    facade._prepress_projection = None

    result = facade.prepress_move_to(
        role="press",
        target_xyz=[0.30, 0.0, 0.60],
        target_quat_xyzw=_z_quat(120.0),
        plan_only=True,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "press_staging_requires_fresh_button_projection"
    assert not any(call[0] == "plan" for call in calls)


def test_press_plan_certificate_binds_fresh_projection(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="button_front_face")
    target = {
        "role": "press",
        "target_xyz": [0.30, 0.0, 0.60],
        "target_quat_xyzw": _z_quat(120.0),
    }
    planned = facade.prepress_move_to(plan_only=True, **target)
    assert planned["primitive_success"] is True
    planner_calls = len([call for call in calls if call[0] == "plan"])

    facade._prepress_projection = {
        **facade._prepress_projection,
        "projection_id": "button_projection_changed",
    }
    result = facade.prepress_move_to(plan_only=False, **target)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "matching_plan_only_certificate_required"
    assert len([call for call in calls if call[0] == "plan"]) == planner_calls


def test_prepress_gate_binding_fails_closed_on_missing_or_expired_frame():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 17
    facade._prepress_gate = {
        "button_visible": True,
        "face_class": BUTTON_FACE_CLASS,
        "gate_id": "button_gate_test",
        "env_step": 17,
        "frame_id": "head:17:frame",
        "camera": "head",
        "resolved_camera": "head",
    }
    facade._frame_cache = SimpleNamespace(
        get_current=lambda *_args: pytest.fail("missing group must not reach cache")
    )
    assert facade._prepress_gate_binding() is None

    facade._prepress_gate["capture_group_id"] = "capture:17"

    def expired(*_args):
        raise CameraGeometryError("expired")

    facade._frame_cache = SimpleNamespace(get_current=expired)
    assert facade._prepress_gate_binding() is None


def test_prepress_three_view_capture_requires_nonempty_group(tmp_path):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 17
    facade._output_dir = tmp_path
    facade._prepress_context = {"held_hand": "right", "press_hand": "left"}
    facade.observe = lambda camera: {
        "camera": camera,
        "frame_id": f"{camera}:17:frame",
        "_image_bytes": b"png",
        "capture_group": {},
    }
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: SimpleNamespace(step_index=17)
    )

    with pytest.raises(RuntimeError, match="omitted capture group"):
        facade._capture_prepress_views(round_index=1)


def test_safe_lift_exception_does_not_cover_large_lateral_motion(tmp_path):
    facade, calls = _motion_guard_facade(tmp_path, face_class="ambiguous")

    _assert_motion_rejected_before_planner(
        facade,
        calls,
        role="held",
        target_xyz=[0.151, 0.0, 0.64],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )


def test_nav_pick_generic_checkpoint_save_cannot_overwrite_checkpoint1_or_bypass_cp2():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    with pytest.raises(RuntimeError, match="immutable"):
        facade.save_robot_state_checkpoint(
            checkpoint_name="state_checkpoint_1",
            stage="post_pi0_nav_pick",
            held_hand="right",
            press_hand="left",
            object_name="radio_89",
        )

    calls = []
    facade.save_prepress_checkpoint = lambda **kwargs: (
        calls.append(kwargs) or {"primitive_success": True}
    )
    result = facade.save_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_2",
        stage="pre_press_alignment",
        held_hand="right",
        press_hand="left",
        object_name="radio_89",
    )
    assert result["primitive_success"] is True
    assert calls == [
        {
            "checkpoint_name": "state_checkpoint_2",
            "stage": "pre_press_alignment",
            "visual_review": True,
        }
    ]


def test_nav_pick_temporary_checkpoint_names_are_bounded_and_non_overwritable(
    tmp_path,
):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    facade._output_dir = tmp_path

    with pytest.raises(ValueError, match="temporary checkpoint stage"):
        facade.save_robot_state_checkpoint(
            checkpoint_name="tmp_state_checkpoint_before_press",
            stage="pre_press_alignment",
            held_hand="right",
            press_hand="left",
            object_name="radio_89",
        )

    checkpoint_dir = tmp_path / "state_checkpoints"
    checkpoint_dir.mkdir()
    existing = checkpoint_dir / "tmp_state_checkpoint_before_press.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable"):
        facade.save_robot_state_checkpoint(
            checkpoint_name="tmp_state_checkpoint_before_press",
            stage="temporary_restore_point",
            held_hand="right",
            press_hand="left",
            object_name="radio_89",
        )

    for index in range(1, 4):
        (checkpoint_dir / f"tmp_state_checkpoint_extra_{index}.json").write_text(
            "{}", encoding="utf-8"
        )
    with pytest.raises(RuntimeError, match="temporary checkpoint limit"):
        facade.save_robot_state_checkpoint(
            checkpoint_name="tmp_state_checkpoint_fifth",
            stage="temporary_restore_point",
            held_hand="right",
            press_hand="left",
            object_name="radio_89",
        )


def test_checkpoint2_finalizer_is_fail_closed_and_does_not_unlink_checkpoint1():
    source = inspect.getsource(BehaviorEnvFacade.save_prepress_checkpoint)
    assert 'checkpoint_name != "state_checkpoint_2"' in source
    assert 'motion.get("visual_review_pass") is True' in source
    assert 'geometry.get("geometry_pass") is True' in source
    assert "path1.unlink" not in source
    assert "_write_json_atomic(path2, payload)" in source


def test_prepress_phase_binds_fresh_frames_checkpoint_and_public_review():
    inspect_source = inspect.getsource(BehaviorEnvFacade.inspect_post_pick_state)
    declare_source = inspect.getsource(BehaviorEnvFacade.declare_button_visibility)
    project_source = inspect.getsource(BehaviorEnvFacade.project_button)
    geometry_source = inspect.getsource(BehaviorEnvFacade.evaluate_prepress_geometry)
    observe_source = inspect.getsource(BehaviorEnvFacade.observe)
    read_source = inspect.getsource(BehaviorEnvFacade._read_post_pick_checkpoint)

    assert "_HANDOFF_PAUSED" in inspect_source
    assert 'self._action_source != "curobo"' in inspect_source
    assert "lifecycle_finalized" in inspect_source
    assert "frame.step_index" in declare_source
    assert "frame.step_index" in project_source
    assert "PREPRESS_LINE_DISTANCE_MAX_M" in geometry_source
    assert "PREPRESS_OPPOSITION_ANGLE_MAX_DEG" in geometry_source
    assert "three_view_observed_by_vlm" in observe_source
    assert "capture_group_ids" in observe_source
    assert "len(capture_group_ids) == 1" in observe_source
    assert "hashlib.sha256(raw).hexdigest()" in read_source

    save_source = inspect.getsource(BehaviorEnvFacade.save_prepress_checkpoint)
    assert 'motion.get("three_view_observed_by_vlm") is True' in save_source


def test_handoff_failure_keeps_local_success_and_persists_step_diagnostics():
    handoff = inspect.getsource(BehaviorEnvFacade._complete_pi0_nav_pick_handoff)
    chunk = inspect.getsource(BehaviorEnvFacade._step_action_chunk)
    assert "handoff_env_steps += 1" in handoff
    assert '"latest_metrics": _wire_safe(latest_metrics)' in handoff
    assert '"handoff_failure_diagnostics_path"' in handoff
    failure_branch = chunk[chunk.index("except Exception as exc:") :]
    assert '"handoff_failed": True' in failure_branch
    assert '"handoff_env_steps": int(' in failure_branch
