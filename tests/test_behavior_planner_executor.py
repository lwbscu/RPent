import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import CameraIntrinsics, FrameCache
from robots.behavior.planner_executor import (
    LOCAL_GUARDED_IK_SEEDS,
    PlannerExecutor,
    RealCuroboBackend,
    _accumulate_actual_dynamics_peak,
    _attachment_identity_status,
    _guarded_waypoint_distances,
    _interpolate_joint_trajectory,
    _quat_to_intrinsic_rpy,
    _retime_joint_trajectory,
    _terminally_smoothed_joint_trajectory,
    _wall_clock_deadline,
)
from robots.behavior.schemas import ENV_ACTION_SEGMENTS
from rpent.tools.toolkit import ToolResult


def test_actual_dynamics_peak_retains_maxima_and_source_joints():
    first = {
        "available": True,
        "ok": True,
        "max_actual_velocity": 0.4,
        "max_actual_velocity_joint": "base_x",
        "max_velocity_ratio": 0.1,
        "max_actual_acceleration": 4.0,
        "max_actual_acceleration_joint": "base_x",
        "max_acceleration_ratio": 0.25,
        "max_velocity_limit": 10.0,
        "max_acceleration_limit": 15.0,
        "sample_dt_s": 1.0 / 60.0,
        "source": "feedback",
    }
    second = {
        **first,
        "max_actual_velocity": 0.2,
        "max_actual_velocity_joint": "arm_1",
        "max_velocity_ratio": 0.2,
        "max_actual_acceleration": 8.0,
        "max_actual_acceleration_joint": "arm_1",
        "max_acceleration_ratio": 0.5,
    }

    peak = _accumulate_actual_dynamics_peak(None, first)
    peak = _accumulate_actual_dynamics_peak(peak, second)

    assert peak["samples"] == 2
    assert peak["max_actual_velocity"] == 0.4
    assert peak["max_actual_velocity_joint"] == "base_x"
    assert peak["max_velocity_ratio"] == 0.2
    assert peak["max_actual_acceleration"] == 8.0
    assert peak["max_actual_acceleration_joint"] == "arm_1"
    assert peak["ok"] is True


class _FakeEnv:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self._last_info = {"done": {"success": False}}
        self._gripper_latch = {"left": 1.0, "right": 1.0}

    def chunk_step(self, actions):
        self.calls.append(np.asarray(actions).copy())
        self.backend.advance()
        return None, 0.0, False, False, self._last_info


class _FakeBackend:
    def __init__(
        self,
        *,
        progress=True,
        bad_actions=False,
        bad_velocity=False,
        reachable=True,
        contact_mode=None,
        attached_obj=None,
        attach_on_close=False,
        assisted_grasp_ray_offset_m=0.04,
    ):
        self.progress = progress
        self.bad_actions = bad_actions
        self.bad_velocity = bad_velocity
        self.reachable = reachable
        self.contact_mode = contact_mode
        self.attached_obj = attached_obj
        self.attach_on_close = attach_on_close
        self.target_root = object()
        self.assisted_grasp_ray_offset_m = assisted_grasp_ray_offset_m
        self.env = None
        self.pose = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.base_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target = None
        self.target_quat = None
        self.planned_targets = []
        self.attached_used = []
        self.hold = (np.arange(23, dtype=np.float32) + 1.0) * 0.01

    def check_arm_reachability(
        self, *, hand, target_xyz, target_quat_xyzw, base_xyyaw=None
    ):
        if not self.reachable:
            return False, "navigation_required", {"eef_target_distance_m": 2.0}
        return (
            True,
            "reachable_candidate",
            {
                "eef_target_distance_m": float(np.linalg.norm(self.pose - target_xyz)),
                "reachability_stage": "world_collision_ik"
                if base_xyyaw is None
                else "candidate_kinematic_ik",
            },
        )

    def plan_arm_trajectory(
        self, *, hand, target_xyz, target_quat_xyzw, timeout_s, attached_obj=None
    ):
        self.target = np.asarray(target_xyz, dtype=np.float64)
        self.target_quat = (
            None
            if target_quat_xyzw is None
            else np.asarray(target_quat_xyzw, dtype=np.float64)
        )
        self.planned_targets.append(self.target.copy())
        self.attached_used.append(attached_obj)
        if self.bad_actions:
            return {"ok": True, "actions": np.zeros((2, 22), dtype=np.float32)}
        if self.bad_velocity:
            actions = np.zeros((3, 23), dtype=np.float32)
            actions[1] = 6.1
            return {"ok": True, "actions": actions}
        return {
            "ok": True,
            "actions": np.zeros((40, 23), dtype=np.float32),
            "metrics": {"trajectory_waypoints": 40},
        }

    def get_eef_pose(self, hand):
        return self.pose.copy(), self.quat.copy()

    def get_base_pose(self):
        return self.base_pose.copy()

    def hold_action(self, hand=None):
        return self.hold.copy()

    def advance(self):
        if self.env is not None:
            for hand in ("left", "right"):
                latch = self.env._gripper_latch[hand]
                eef_link = f"{hand}_eef_link"
                if latch > 0:
                    if isinstance(self.attached_obj, dict):
                        if eef_link in self.attached_obj:
                            self.attached_obj = None
                elif self.attach_on_close and self.attached_obj is None:
                    self.attached_obj = {eef_link: self.target_root}
        if self.progress and self.target is not None:
            self.pose = self.pose + 0.5 * (self.target - self.pose)
            if self.target_quat is not None:
                self.quat = self.target_quat.copy()

    def joint_margin(self):
        return 0.5

    def joint_margin_report(self):
        return {
            "available": True,
            "min_normalized_margin": 0.5,
            "threshold_normalized": 0.03,
            "threshold_raw_rad": 0.05,
            "ok": True,
        }

    def collision_report(self):
        return {
            "available": True,
            "colliding": False,
            "min_margin_m": 0.02,
            "margin_available": True,
        }

    def contact_report(
        self, *, hand, target_xyz=None, allowed_contact_distance_m=0.025
    ):
        if self.contact_mode == "unexpected":
            return {
                "available": True,
                "contact_count": 1,
                "unexpected_contact": True,
                "expected_contact": False,
                "allowed_contact_distance_m": allowed_contact_distance_m,
            }
        if self.contact_mode == "unavailable":
            return {
                "available": False,
                "reason": "fake_contact_api_unavailable",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        if self.contact_mode == "expected":
            return {
                "available": True,
                "contact_count": 1,
                "unexpected_contact": False,
                "expected_contact": True,
                "allowed_contact_distance_m": allowed_contact_distance_m,
            }
        if self.contact_mode == "expected_near_target":
            expected = bool(
                target_xyz is not None
                and np.linalg.norm(self.pose - np.asarray(target_xyz, dtype=np.float64))
                <= allowed_contact_distance_m
            )
            return {
                "available": True,
                "contact_count": int(expected),
                "unexpected_contact": False,
                "expected_contact": expected,
                "allowed_contact_distance_m": allowed_contact_distance_m,
            }
        return {
            "available": True,
            "contact_count": 0,
            "unexpected_contact": False,
            "expected_contact": False,
            "allowed_contact_distance_m": allowed_contact_distance_m,
        }

    def guarded_contact_safety_report(
        self,
        *,
        hand,
        target_xyz,
        allowed_contact_distance_m=0.025,
    ):
        contact = self.contact_report(
            hand=hand,
            target_xyz=target_xyz,
            allowed_contact_distance_m=allowed_contact_distance_m,
        )
        return {
            "available": contact.get("available", False),
            "target_object_resolved": self.contact_mode != "target_unresolved",
            "self_colliding": self.contact_mode == "self_collision",
            "world_without_target_colliding": self.contact_mode == "unrelated_world",
            "target_only_activation_verified": self.contact_mode
            not in {"self_collision", "unrelated_world", "zero_unverified"},
            "contact_report": contact,
        }

    def get_attached_object(self, hand):
        return self.attached_obj

    def resolve_target_attachment(self, *, hand, target_xyz):
        del target_xyz
        return {f"{hand}_eef_link": self.target_root}

    def get_assisted_grasp_outward_ray_geometry(self, hand):
        assert hand in {"left", "right"}
        offset = self.assisted_grasp_ray_offset_m
        if offset is None:
            return None
        return {
            "available": True,
            "outward_offset_m": float(offset),
            "start_outward_offset_m": float(offset),
            "end_outward_offset_m": float(offset),
            "plane_mismatch_m": 0.0,
            "ray_span_m": 0.04,
            "start_point_count": 2,
            "end_point_count": 2,
            "frame": "eef_local_positive_z",
        }

    def clear_attached_object(self, hand):
        self.attached_obj = None

    def joint_tracking_report(self, target_q, *, hand):
        return {
            "available": True,
            "reached": True,
            "max_articulation_error_rad": 0.0,
            "max_base_xy_error_m": 0.0,
            "base_yaw_error_rad": 0.0,
        }

    def capture_locked_joint_reference(self, *, hand):
        return {"hand": hand, "token": "fixed_at_trajectory_start"}

    def locked_joint_drift_report(self, *, reference):
        assert reference["token"] == "fixed_at_trajectory_start"
        return {
            "available": True,
            "ok": True,
            "base_xy_drift_m": 0.0,
            "base_xy_threshold_m": 0.01,
            "base_z_drift_m": 0.0,
            "base_z_threshold_m": 0.01,
            "base_rpy_drift_rad": 0.0,
            "base_rpy_threshold_rad": np.deg2rad(1.0),
            "articulation_drift_rad": 0.0,
            "articulation_threshold_rad": 0.01,
            "locked_joint_count": 1,
        }


def _executor(backend, *, output_dir=None):
    cache = FrameCache()
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0, width=5, height=5)
    cache.add(
        camera="head",
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=np.ones((5, 5), dtype=np.float32),
        intrinsics=intr,
        camera_to_world=np.eye(4),
        step_index=0,
        frame_id="f0",
    )
    env = _FakeEnv(backend)
    backend.env = env
    return (
        PlannerExecutor(
            env=env,
            frame_cache=cache,
            backend=backend,
            output_dir=output_dir,
        ),
        env,
    )


def test_pixel_to_world_returns_planner_result_without_task_success():
    executor, _ = _executor(_FakeBackend())

    result = executor.pixel_to_world(
        camera="head",
        frame_id="f0",
        u=2,
        v=2,
        depth_window_px=3,
    )

    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "projected"
    assert result["suggested_next_tool"] == "move_to"
    np.testing.assert_allclose(result["diagnostics"]["xyz"], [0.0, 0.0, -1.0])


def test_every_planner_call_persists_complete_result_artifact(tmp_path):
    executor, _ = _executor(_FakeBackend(), output_dir=tmp_path)

    result = executor.rotate_wrist(
        hand="left",
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        relative_axis_angle=[0.0, 0.0, 1.0, 0.1],
    )

    artifact = (
        tmp_path / "planner_tool_artifacts" / result["tool_artifact"].split("/")[-1]
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["tool"] == "rotate_wrist"
    assert payload["result"]["primitive_success"] is False
    assert payload["result"]["task_success"] is False
    assert payload["result"]["stop_reason"] == "error"


def test_move_to_executes_23d_actions_until_target_is_held():
    executor, env = _executor(_FakeBackend(progress=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "reached"
    assert result["metrics"]["final_position_error_m"] <= 0.02
    assert env.calls
    assert all(call.shape == (1, 23) for call in env.calls)


def test_move_to_unreachable_suggests_navigation_without_stepping_env():
    executor, env = _executor(_FakeBackend(reachable=False))

    result = executor.move_to(hand="right", target_xyz=[2.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_required"
    assert result["suggested_next_tool"] == "navigate_to"
    assert env.calls == []


def test_base_planner_tries_next_ranked_station_after_curobo_failure():
    class _BaseBackend(RealCuroboBackend):
        def __init__(self):
            self.targets = []

        def _find_robot(self):
            return object()

        def _base_xy_yaw(self, robot):
            return np.array([0.0, 0.0, 0.0, 0.0])

        def _ranked_base_candidates(
            self, robot, *, hand, target_xyz, standoff_m, deadline
        ):
            assert deadline > 0
            return [
                {
                    "xyyaw": np.array([1.0, 0.0, 0.0]),
                    "geodesic_distance_m": 1.0,
                    "reachability_target_xyz": [1.9, 0.0, 1.0],
                    "reachability_reason": "reachable_candidate",
                    "reachability_stage": "candidate_kinematic_ik",
                },
                {
                    "xyyaw": np.array([0.0, 1.0, 1.57]),
                    "geodesic_distance_m": 1.2,
                    "reachability_target_xyz": [1.9, 0.0, 1.0],
                    "reachability_reason": "reachable_candidate",
                    "reachability_stage": "candidate_kinematic_ik",
                },
            ]

        def _compute_base_plan(
            self, *, target_xyyaw, timeout_s, skip_obstacle_update=False
        ):
            assert skip_obstacle_update is True
            self.targets.append(target_xyyaw.copy())
            if len(self.targets) == 1:
                return {
                    "ok": False,
                    "stop_reason": "base_plan_failed",
                    "metrics": {"successes": [False]},
                }
            return {
                "ok": True,
                "joint_trajectory": np.zeros((2, 28), dtype=np.float32),
                "metrics": {"successes": [True]},
            }

    backend = _BaseBackend()

    result = backend.plan_base_trajectory(
        hand="left",
        target_xyz=np.array([2.0, 0.0, 1.0]),
        standoff_m=0.85,
        timeout_s=10.0,
    )

    assert result["ok"] is True
    assert len(backend.targets) == 2
    assert result["base_goal"] == [0.0, 1.0, 1.57]
    assert [attempt["ok"] for attempt in result["metrics"]["base_plan_attempts"]] == [
        False,
        True,
    ]


def test_base_planner_distinguishes_collision_only_failure():
    class Backend(RealCuroboBackend):
        def __init__(self):
            super().__init__(None)

        def _find_robot(self):
            return object()

        def _base_xy_yaw(self, _robot):
            return np.zeros(4)

        def _ranked_base_candidates(self, *_args, **_kwargs):
            self._last_base_candidate_summary = {
                "traversable_count": 3,
                "collision_free_count": 0,
                "collision_colliding_count": 3,
                "collision_unavailable_count": 0,
            }
            return []

    result = Backend().plan_base_trajectory(
        hand="left",
        target_xyz=np.array([1.0, 0.0, 0.8]),
        standoff_m=0.85,
        timeout_s=1.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "navigation_collision"
    assert result["metrics"]["candidate_summary"]["traversable_count"] == 3


def test_base_planner_does_not_misreport_unavailable_collision_check():
    class Backend(RealCuroboBackend):
        def __init__(self):
            super().__init__(None)

        def _find_robot(self):
            return object()

        def _base_xy_yaw(self, _robot):
            return np.zeros(4)

        def _ranked_base_candidates(self, *_args, **_kwargs):
            self._last_base_candidate_summary = {
                "traversable_count": 3,
                "collision_free_count": 0,
                "collision_colliding_count": 0,
                "collision_unavailable_count": 3,
            }
            return []

    result = Backend().plan_base_trajectory(
        hand="left",
        target_xyz=np.array([1.0, 0.0, 0.8]),
        standoff_m=0.85,
        timeout_s=1.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "planner_unavailable"


def test_move_to_rejects_bad_action_shape_before_env_step():
    executor, env = _executor(_FakeBackend(bad_actions=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "ValueError" in result["diagnostics"]["error"]
    assert env.calls == []


def test_move_to_stops_on_stalled_tracking():
    executor, env = _executor(_FakeBackend(progress=False))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0], timeout_s=10)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "stalled_tracking"
    assert result["recoverable"] is True
    assert 20 <= len(env.calls) <= 25


def test_held_move_uses_one_captured_collision_body_for_reachability_and_plan():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__(attached_obj={"left_eef_link": self_root})
            self.attachment_reads = 0
            self.reachability_attached = []

        def get_attached_object(self, hand):
            assert hand == "left"
            self.attachment_reads += 1
            return self.attached_obj

        def check_arm_reachability(
            self,
            *,
            hand,
            target_xyz,
            target_quat_xyzw,
            attached_obj,
            timeout_s,
        ):
            del timeout_s
            self.reachability_attached.append(attached_obj)
            return super().check_arm_reachability(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
            )

    self_root = object()
    backend = Backend()
    captured = backend.attached_obj
    executor, _ = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.5, 0.0, 0.0],
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert backend.attachment_reads == 1
    assert backend.reachability_attached == [captured]
    assert backend.reachability_attached[0] is captured
    assert backend.attached_used[-1] is captured
    assert result["metrics"]["attached_collision_body"] == {
        "available": True,
        "identity_locked_at_call_start": True,
        "used_for_reachability": True,
        "used_for_full_trajectory_plan": True,
        "required_during_execution": True,
    }


def test_held_move_validates_exact_attachment_identity_after_every_step():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"left_eef_link": self.target_root}
            self.attachment_reads = 0

        def get_attached_object(self, hand):
            assert hand == "left"
            self.attachment_reads += 1
            if self.attached_obj is None:
                return None
            # The mapping may be reconstructed by OG, but the exact root body
            # must remain identical.
            return {"left_eef_link": self.attached_obj["left_eef_link"]}

    backend = Backend()
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0

    result = executor.move_to(hand="left", target_xyz=[0.5, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert backend.attachment_reads >= len(env.calls) + 2
    trace = result["diagnostics"]["trace"]
    assert trace
    assert all(sample["attachment_identity"]["matches"] for sample in trace)


def test_held_move_fails_bounded_when_attachment_is_lost():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"left_eef_link": self.target_root}
            self.advance_calls = 0

        def advance(self):
            self.advance_calls += 1
            super().advance()
            if self.advance_calls == 2:
                self.attached_obj = None

    backend = Backend()
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0

    result = executor.move_to(hand="left", target_xyz=[0.4, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_lost"
    assert len(env.calls) == 2
    assert (
        result["metrics"]["attached_collision_body"]["required_during_execution"]
        is True
    )


def test_held_move_fails_bounded_on_attachment_identity_mismatch():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"right_eef_link": self.target_root}

    backend = Backend()
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0

    result = executor.move_to(hand="left", target_xyz=[0.4, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_identity_mismatch"
    assert env.calls == []


def test_observe_payload_becomes_tool_result_image_block_without_pixel_dump():
    executor, _ = _executor(_FakeBackend())

    payload = executor.observe("head")
    result = ToolResult(name="observe", result=payload)

    assert payload["_image_bytes"].startswith(b"\x89PNG")
    assert "rgb" not in payload
    assert any(block["type"] == "image" for block in result.content_blocks)
    assert "_image_bytes" not in result.content_blocks[0]["text"]
    assert "[[[" not in result.content_blocks[0]["text"]


def test_rotate_wrist_requires_four_value_axis_angle_and_xor():
    backend = _FakeBackend()
    executor, _ = _executor(backend)

    both = executor.rotate_wrist(
        hand="left",
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        relative_axis_angle=[0.0, 0.0, 1.0, 0.1],
    )
    bad = executor.rotate_wrist(hand="left", relative_axis_angle=[0.0, 0.0, 0.1])
    ok = executor.rotate_wrist(
        hand="left", relative_axis_angle=[0.0, 0.0, 1.0, np.pi / 8]
    )

    assert both["primitive_success"] is False
    assert "exactly one" in both["diagnostics"]["error"]
    assert bad["primitive_success"] is False
    assert "axis_x" in bad["diagnostics"]["error"]
    assert ok["primitive_success"] is True


def test_press_uses_guarded_two_millimeter_waypoints():
    backend = _FakeBackend(contact_mode="expected_near_target")
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        approach_distance_m=0.004,
        press_depth_m=0.004,
    )

    assert result["primitive_success"] is True
    guarded_targets = backend.planned_targets[1:]
    assert guarded_targets
    max_step = max(
        np.linalg.norm(guarded_targets[i] - guarded_targets[i - 1])
        for i in range(1, len(guarded_targets))
    )
    assert max_step <= 0.002 + 1e-9
    assert result["metrics"]["guarded_step_m"] == 0.002


def test_default_press_precontact_stays_outside_curobo_activation_zone():
    backend = _FakeBackend(contact_mode="expected_near_target")
    executor, _ = _executor(backend)
    surface = np.array([0.5, 0.0, -0.1])

    result = executor.press(
        hand="left",
        target_xyz=surface,
        press_direction=[0.0, 0.0, -1.0],
    )

    assert result["primitive_success"] is True
    precontact = backend.planned_targets[0]
    assert precontact[2] - surface[2] >= 0.006 - 1e-9


def test_eight_centimeter_guarded_path_is_two_millimeters_throughout():
    distances = np.asarray(_guarded_waypoint_distances(0.08), dtype=np.float64)
    increments = np.diff(np.r_[0.0, distances])

    assert distances[-1] == pytest.approx(0.08)
    assert np.max(increments) <= 0.002 + 1e-12


def test_guarded_terminal_smoothing_preserves_line_and_reaches_zero_velocity():
    alpha = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    trajectory = np.column_stack([0.096 * alpha, -0.048 * alpha])

    smoothed, report = _terminally_smoothed_joint_trajectory(trajectory)

    np.testing.assert_allclose(smoothed[0], trajectory[0])
    np.testing.assert_allclose(smoothed[-1], trajectory[-1])
    nonzero_x = smoothed[:, 0] > 1e-10
    np.testing.assert_allclose(
        smoothed[nonzero_x, 1] / smoothed[nonzero_x, 0],
        -0.5,
        atol=1e-6,
    )
    deltas = np.diff(smoothed, axis=0)
    np.testing.assert_allclose(deltas[-3:], 0.0, atol=1e-8)
    eased_steps = report["terminal_ease_out_steps"]
    eased_norms = np.max(np.abs(deltas[-(eased_steps + 3) : -3]), axis=1)
    assert np.all(np.diff(eased_norms) <= 1e-7)
    assert report["path_geometry"] == "original_joint_polyline"
    assert report["full_collision_recheck_required"] is True
    assert report["terminal_max_command_acceleration_proxy_rad_s2"] < 15.0


def test_real_style_certified_guarded_planner_batches_two_mm_targets_once():
    class CertifiedBackend(_FakeBackend):
        def plan_guarded_ik_path(
            self,
            *,
            hand,
            target_xyz,
            target_quat_xyzw,
            timeout_s,
            attached_obj=None,
            contact_target_xyz=None,
        ):
            assert contact_target_xyz is not None
            return self.plan_arm_trajectory(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                attached_obj=attached_obj,
            )

    backend = CertifiedBackend(contact_mode="expected_near_target")
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        approach_distance_m=0.004,
        press_depth_m=0.004,
    )

    assert result["primitive_success"] is True
    guarded_targets = backend.planned_targets[1:]
    assert len(guarded_targets) == 1
    assert result["metrics"]["guarded_execution_mode"].startswith("single_batch_curobo")


def test_guarded_contact_aborts_before_target_neighborhood():
    backend = _FakeBackend(contact_mode="expected")
    executor, env = _executor(backend)

    result = executor._guarded_incremental_move(
        hand="left",
        target_xyz=np.array([0.5, 0.0, -0.04]),
        target_quat_xyzw=None,
        direction=np.array([0.0, 0.0, -1.0]),
        allow_expected_contact=True,
        position_tolerance_m=0.015,
        timeout_s=45.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "unexpected_contact"
    assert len(env.calls) < 60


def test_guarded_contact_api_unavailable_is_structured_failure():
    backend = _FakeBackend(contact_mode="unavailable")
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.01],
        press_direction=[0.0, 0.0, -1.0],
        approach_distance_m=0.004,
        press_depth_m=0.004,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "contact_feedback_unavailable"
    assert result["metrics"]["contact_report"]["available"] is False


def test_guarded_world_collision_exception_is_local_and_contact_checked():
    target = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    safe_backend = _FakeBackend(contact_mode="expected_near_target")
    safe, _ = _executor(safe_backend)

    allowed, report = safe._guarded_target_collision_allowed(
        hand="left",
        target_xyz=target,
        enabled=True,
    )

    assert allowed is True
    assert report["exception_scope"].endswith("self_never_exempt")
    unsafe_backend = _FakeBackend(contact_mode="unexpected")
    unsafe, _ = _executor(unsafe_backend)
    rejected, unsafe_report = unsafe._guarded_target_collision_allowed(
        hand="left",
        target_xyz=target,
        enabled=True,
    )
    assert rejected is False
    assert unsafe_report["contact_report"]["unexpected_contact"] is True
    safe_backend.pose = np.array([0.6, 0.0, 0.0], dtype=np.float64)
    far, far_report = safe._guarded_target_collision_allowed(
        hand="left",
        target_xyz=target,
        enabled=True,
    )
    assert far is False
    assert far_report["eef_target_distance_m"] > 0.025


def test_guarded_contact_neighborhood_uses_fingertip_not_eef_origin():
    target = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    backend = _FakeBackend(contact_mode="expected")
    backend.pose = target + np.array([0.0, 0.0, 0.04])
    executor, _ = _executor(backend)
    contact = backend.contact_report(
        hand="left", target_xyz=target, allowed_contact_distance_m=0.025
    )

    assert (
        executor._contact_is_abort(
            contact,
            hand="left",
            target_xyz=target,
            allow_expected_contact=True,
        )
        is True
    )
    assert (
        executor._contact_is_abort(
            contact,
            hand="left",
            target_xyz=target,
            allow_expected_contact=True,
            eef_to_contact_vector=np.array([0.0, 0.0, -0.04]),
        )
        is False
    )
    allowed, report = executor._guarded_target_collision_allowed(
        hand="left",
        target_xyz=target,
        enabled=True,
        eef_to_contact_vector=np.array([0.0, 0.0, -0.04]),
    )
    assert allowed is True
    assert report["eef_contact_target_distance_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "contact_mode",
    ["zero_unverified", "self_collision", "unrelated_world", "target_unresolved"],
)
def test_guarded_world_collision_never_exempts_unverified_or_non_target_hazards(
    contact_mode,
):
    backend = _FakeBackend(contact_mode=contact_mode)
    executor, _ = _executor(backend)

    allowed, report = executor._guarded_target_collision_allowed(
        hand="left",
        target_xyz=np.array([0.5, 0.0, 0.0]),
        enabled=True,
    )

    assert allowed is False
    assert report["world_collision_exception_allowed"] is False


def test_move_to_executes_hold_until_ten_stable_steps():
    executor, env = _executor(_FakeBackend(progress=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert result["metrics"]["held_steps"] >= 10
    assert len(env.calls) > 10


def test_move_to_rejects_velocity_limit_before_env_step():
    executor, env = _executor(_FakeBackend(bad_velocity=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "velocity_limit"
    assert env.calls == []


def test_move_to_fails_closed_without_collision_feedback():
    backend = _FakeBackend()
    backend.collision_report = lambda: {
        "available": False,
        "reason": "fake_collision_unavailable",
    }
    executor, env = _executor(backend)

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "collision_feedback_unavailable"
    assert env.calls == []


def test_move_to_fails_closed_without_joint_limit_feedback():
    backend = _FakeBackend()
    backend.joint_margin_report = lambda: {
        "available": False,
        "reason": "fake_joint_feedback_unavailable",
        "ok": None,
    }
    executor, env = _executor(backend)

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "joint_limit_feedback_unavailable"
    assert env.calls == []


def test_move_to_forces_post_step_collision_recheck():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.collision_forces = []

        def collision_report(self, *, force=False):
            self.collision_forces.append(force)
            return {
                "available": True,
                "colliding": False,
                "min_margin_m": 0.02,
                "margin_available": True,
            }

    backend = Backend()
    executor, env = _executor(backend)

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert backend.collision_forces.count(True) == len(env.calls)
    assert backend.collision_forces.count(False) >= len(env.calls)


def test_public_planner_timeout_is_not_swallowed_by_collision_feedback():
    class Backend(_FakeBackend):
        def collision_report(self, *, force=False):
            del force
            time.sleep(0.05)
            return {
                "available": True,
                "colliding": False,
                "min_margin_m": 0.005,
                "margin_available": True,
            }

    executor, _ = _executor(Backend())
    started = time.monotonic()
    result = executor.move_to(
        hand="left",
        target_xyz=[0.0, 0.0, 0.0],
        timeout_s=0.01,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "timeout"
    assert time.monotonic() - started < 0.1


def test_real_collision_feedback_checks_each_step_and_refreshes_world_periodically(
    tmp_path,
):
    class Env:
        _env_steps = 0

    class Robot:
        _ag_obj_in_hand = {}

        @staticmethod
        def get_joint_positions():
            return np.zeros(28, dtype=np.float32)

    class Generator:
        def __init__(self):
            self.skip_obstacle_updates = []

        def check_collisions(
            self,
            q,
            *,
            self_collision_check,
            skip_obstacle_update,
            attached_obj,
        ):
            assert self_collision_check is True
            assert attached_obj is None
            self.skip_obstacle_updates.append(skip_obstacle_update)
            return np.zeros(len(q), dtype=bool)

    env = Env()
    robot = Robot()
    generator = Generator()
    backend = RealCuroboBackend(env, output_dir=tmp_path)
    backend._robot = robot
    backend._active_generator = generator
    assert backend.collision_report(force=True)["obstacle_world_refreshed"] is True
    env._env_steps = 1
    assert backend.collision_report(force=True)["obstacle_world_refreshed"] is False
    env._env_steps = 4
    assert backend.collision_report(force=True)["obstacle_world_refreshed"] is True
    assert generator.skip_obstacle_updates == [False, True, False]


def test_guarded_target_exclusion_combines_independent_world_and_self_checks(tmp_path):
    target_object = object()

    class Generator:
        def __init__(self):
            self.obstacle_updates = []
            self.collision_modes = []

        def update_obstacles(self, ignore_objects=None):
            self.obstacle_updates.append(ignore_objects)

        def check_collisions(
            self,
            q,
            *,
            self_collision_check,
            skip_obstacle_update,
            attached_obj,
        ):
            assert np.asarray(q).shape == (1, 28)
            assert skip_obstacle_update is True
            assert attached_obj is None
            self.collision_modes.append(self_collision_check)
            return np.array([False])

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._target_object_for_point = lambda _target: target_object
    backend._check_q_self_collisions = lambda *_args, **_kwargs: {
        "available": True,
        "colliding": False,
        "colliding_mask": [False],
    }

    report = backend._check_q_target_excluded_collisions(
        generator,
        np.zeros((1, 28), dtype=np.float32),
        target_xyz=np.zeros(3, dtype=np.float64),
    )

    assert report["available"] is True
    assert report["colliding"] is False
    assert report["target_only_activation_verified"] is True
    assert generator.collision_modes == [False]
    assert generator.obstacle_updates == [[target_object], None]


def test_guarded_without_contact_target_uses_combined_world_and_self_check(tmp_path):
    class Generator:
        def __init__(self):
            self.calls = []

        def check_collisions(
            self,
            q,
            *,
            self_collision_check,
            skip_obstacle_update,
            attached_obj,
        ):
            self.calls.append(
                (self_collision_check, skip_obstacle_update, attached_obj)
            )
            return np.zeros(len(q), dtype=bool)

    attached = object()
    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)

    report = backend._check_q_combined_collisions(
        generator,
        np.zeros((2, 28), dtype=np.float32),
        attached_obj=attached,
    )

    assert report["available"] is True
    assert report["colliding"] is False
    assert report["checked_waypoints"] == 2
    assert report["collision_scope"] == "combined_world_and_self"
    assert generator.calls == [(True, False, attached)]


def test_guarded_without_contact_target_fails_closed_when_check_raises(tmp_path):
    class Generator:
        @staticmethod
        def check_collisions(*_args, **_kwargs):
            raise RuntimeError("collision backend unavailable")

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    report = backend._check_q_combined_collisions(
        Generator(), np.zeros((1, 28), dtype=np.float32)
    )

    assert report["available"] is False
    assert report["collision_scope"] == "combined_world_and_self"
    assert "collision backend unavailable" in report["reason"]


def test_guarded_target_exclusion_fails_closed_on_independent_self_collision(
    tmp_path,
):
    class Generator:
        def __init__(self):
            self.restored = False

        def update_obstacles(self, ignore_objects=None):
            self.restored = ignore_objects is None

        def check_collisions(
            self,
            q,
            *,
            self_collision_check,
            skip_obstacle_update,
            attached_obj,
        ):
            return np.array([bool(self_collision_check)])

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._target_object_for_point = lambda _target: object()
    backend._check_q_self_collisions = lambda *_args, **_kwargs: {
        "available": True,
        "colliding": True,
        "colliding_mask": [True],
    }

    report = backend._check_q_target_excluded_collisions(
        generator,
        np.zeros((1, 28), dtype=np.float32),
        target_xyz=np.zeros(3, dtype=np.float64),
    )

    assert report["available"] is True
    assert report["colliding"] is True
    assert report["self_or_unrelated_world_colliding"] is True
    assert report["target_only_activation_verified"] is False
    assert generator.restored is True


def test_ik_solution_merge_preserves_every_locked_joint(tmp_path):
    class Robot:
        @staticmethod
        def get_joint_positions():
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    lock_state = type("LockState", (), {"joint_names": ["locked_b", "locked_d"]})()
    config = type("Config", (), {"lock_jointstate": lock_state})()
    kinematics = type("Kinematics", (), {"kinematics_config": config})()
    motion_generator = type("MotionGenerator", (), {"kinematics": kinematics})()
    embodiment = type("Embodiment", (), {"DEFAULT": "default"})
    generator = type(
        "Generator",
        (),
        {
            "robot_joint_names": ["active_a", "locked_b", "active_c", "locked_d"],
            "mg": {"default": motion_generator},
        },
    )()
    path = type(
        "Path",
        (),
        {
            "joint_names": ["active_a", "locked_b", "active_c", "locked_d"],
            "position": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        },
    )()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._embodiment_cls = embodiment

    merged, report = backend._merge_ik_solution_into_full_q(generator, Robot(), path)

    assert merged.tolist() == [[10.0, 2.0, 30.0, 4.0]]
    assert report["active_joint_count"] == 2
    assert report["locked_solution_entries_ignored"] == 2


def test_target_object_resolution_supports_scene_object_mapping(tmp_path):
    target_object = type(
        "Target",
        (),
        {
            "visual_only": False,
            "get_base_aligned_bbox": staticmethod(
                lambda **_kwargs: (
                    np.array([1.0, 2.0, 3.0]),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    np.array([0.1, 0.1, 0.1]),
                    np.zeros(3),
                )
            ),
        },
    )()
    scene = type("Scene", (), {"objects": {"target": target_object}})()
    robot = type("Robot", (), {"scene": scene})()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._robot = robot

    resolved = backend._target_object_for_point(np.array([1.0, 2.0, 3.0]))

    assert resolved is target_object


def test_target_object_resolution_fails_closed_on_overlapping_bbox_ambiguity(tmp_path):
    def make_object(name):
        return type(
            name,
            (),
            {
                "visual_only": False,
                "get_base_aligned_bbox": staticmethod(
                    lambda **_kwargs: (
                        np.array([1.0, 2.0, 3.0]),
                        np.array([0.0, 0.0, 0.0, 1.0]),
                        np.array([0.1, 0.1, 0.1]),
                        np.zeros(3),
                    )
                ),
            },
        )()

    objects = {"first": make_object("First"), "second": make_object("Second")}
    scene = type("Scene", (), {"objects": objects})()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._robot = type("Robot", (), {"scene": scene})()

    assert backend._target_object_for_point(np.array([1.0, 2.0, 3.0])) is None


def test_real_contact_requires_target_identity_and_contact_point_neighborhood(tmp_path):
    target = type(
        "Target",
        (),
        {
            "visual_only": False,
            "prim_path": "/World/target",
            "get_base_aligned_bbox": staticmethod(
                lambda **_kwargs: (
                    np.zeros(3),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    np.array([0.1, 0.1, 0.1]),
                    np.zeros(3),
                )
            ),
        },
    )()

    class Robot:
        scene = type("Scene", (), {"objects": {"target": target}})()
        contacts = []

        @classmethod
        def _find_gripper_contacts(cls, **_kwargs):
            return cls.contacts, []

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._robot = Robot()
    Robot.contacts = [("/World/target/link", np.array([0.01, 0.0, 0.0]))]
    near = backend.contact_report(
        hand="left",
        target_xyz=np.zeros(3),
        allowed_contact_distance_m=0.025,
    )
    Robot.contacts = [("/World/target/link", np.array([0.03, 0.0, 0.0]))]
    far = backend.contact_report(
        hand="left",
        target_xyz=np.zeros(3),
        allowed_contact_distance_m=0.025,
    )

    assert near["expected_contact"] is True
    assert near["unexpected_contact"] is False
    assert far["expected_contact"] is False
    assert far["unexpected_contact"] is True
    assert far["far_target_contact_count"] == 1


def test_quarantined_generator_is_rebuilt_and_warmed_before_reuse(tmp_path):
    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    old = object()
    backend._generators["arm:left"] = old
    backend._active_generator = old
    quarantined = backend._quarantine_generator(
        kind="arm", hand="left", reason="TimeoutError: injected"
    )
    warmed = []
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = type("Embodiment", (), {"DEFAULT": "default"})
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand: tmp_path / "official_arm.yaml"
    backend._probe_generator_lock_resolution = lambda *_args, **_kwargs: None
    backend._warmup_rebuilt_generator = lambda **kwargs: warmed.append(
        (kwargs["kind"], kwargs["hand"])
    )

    rebuilt = backend._generator(kind="arm", hand="left")

    assert quarantined["requires_rebuild_and_warmup"] is True
    assert rebuilt is not old
    assert warmed == [("arm", "left")]
    assert "arm:left" not in backend._invalid_generators


def test_generator_recovery_failure_remains_quarantined(tmp_path):
    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._invalid_generators.add("arm:left")
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = type("Embodiment", (), {"DEFAULT": "default"})
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand: tmp_path / "official_arm.yaml"
    backend._probe_generator_lock_resolution = lambda *_args, **_kwargs: None
    backend._warmup_rebuilt_generator = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("unsafe recovery")
    )

    with pytest.raises(RuntimeError, match="recovery failed closed"):
        backend._generator(kind="arm", hand="left")

    assert "arm:left" in backend._invalid_generators
    assert "arm:left" not in backend._generators


def test_guarded_ik_uses_current_state_as_public_solver_seed_and_retract(tmp_path):
    torch = pytest.importorskip("torch")

    class Solver:
        def __init__(self):
            self.kwargs = []

        def solve_single(self, _goal_pose, **kwargs):
            self.kwargs.append(kwargs)
            return type(
                "Result",
                (),
                {
                    "success": torch.tensor([[True, True]]),
                    "error": torch.tensor([[0.01, 0.2]]),
                    "js_solution": torch.tensor(
                        [[[3.1, 4.1], [1.1, 2.1]]],
                        dtype=torch.float32,
                    ),
                },
            )()

    solver = Solver()
    generator = type(
        "Generator",
        (),
        {"mg": {"default": type("MG", (), {"ik_solver": solver})()}},
    )()
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    start_state = type("Start", (), {"position": current})()
    goal_pose = type(
        "GoalPose",
        (),
        {"batch": 2, "__getitem__": lambda self, _index: object()},
    )()
    backend = RealCuroboBackend(None, output_dir=tmp_path)

    _result, success, paths = backend._solve_local_seeded_ik_batch(
        generator,
        start_state,
        goal_pose,
        object(),
        link_poses=None,
        emb_sel="default",
    )

    assert success.tolist() == [True, True]
    assert np.allclose(
        np.asarray([path.tolist() for path in paths]),
        np.asarray([[1.1, 2.1], [1.1, 2.1]]),
    )
    assert torch.equal(solver.kwargs[0]["retract_config"], current[0:1])
    first_seeds = solver.kwargs[0]["seed_config"]
    assert first_seeds.shape == (1, LOCAL_GUARDED_IK_SEEDS, 2)
    assert torch.equal(first_seeds[:, 0:1], current[0:1].unsqueeze(1))
    assert float((first_seeds[0] - current[0]).abs().max()) <= 0.02
    assert torch.equal(
        solver.kwargs[1]["retract_config"],
        torch.tensor([[1.1, 2.1]], dtype=torch.float32),
    )
    assert solver.kwargs[0]["return_seeds"] == LOCAL_GUARDED_IK_SEEDS
    assert solver.kwargs[0]["num_seeds"] == LOCAL_GUARDED_IK_SEEDS


def test_rotate_wrist_passes_attached_body_to_planner_without_exposing_pose():
    attached = {"left_eef_link": object()}
    backend = _FakeBackend(attached_obj=attached)
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0

    result = executor.rotate_wrist(
        hand="left", relative_axis_angle=[0.0, 0.0, 1.0, 0.1]
    )

    assert result["primitive_success"] is True
    assert attached in backend.attached_used
    assert "object at" not in str(result["diagnostics"])


def test_real_backend_wraps_assisted_grasp_object_by_selected_eef_link():
    root_link = object()

    class Attached:
        pass

    attached = Attached()
    attached.root_link = root_link

    class Robot:
        _ag_obj_in_hand = {"left": attached, "right": None}

    backend = RealCuroboBackend(None)
    backend._robot = Robot()

    assert backend.get_attached_object("left") == {"left_eef_link": root_link}
    assert backend.get_attached_object("right") is None


def test_attachment_identity_metrics_do_not_expose_prim_paths():
    root = SimpleNamespace(prim_path="/World/private/radio/root_link")

    matches, report = _attachment_identity_status(
        {"right_eef_link": root},
        {"right_eef_link": root},
        hand="right",
    )

    assert matches is True
    assert report == {
        "expected_available": True,
        "actual_available": True,
        "identity_kind": "prim_path",
        "matches": True,
    }
    assert "/World" not in str(report)


def test_real_backend_resolves_hand_specific_ag_ray_offsets_in_eef_frame():
    class Link:
        def __init__(self, position, quat):
            self.position = np.asarray(position, dtype=np.float64)
            self.quat = np.asarray(quat, dtype=np.float64)

        def get_position_orientation(self):
            return self.position, self.quat

    world_position = [1.2, -0.7, 0.9]
    world_quat = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    links = {}
    starts = {}
    ends = {}
    for hand, start_offset, end_offset in (
        ("left", 0.015, 0.016),
        ("right", 0.012, 0.012),
    ):
        start_link = f"{hand}_finger_a"
        end_link = f"{hand}_finger_b"
        links[f"{hand}_eef_link"] = Link(world_position, world_quat)
        links[start_link] = Link(world_position, world_quat)
        links[end_link] = Link(world_position, world_quat)
        starts[hand] = [
            SimpleNamespace(link_name=start_link, position=[0.0, -0.02, -start_offset]),
            SimpleNamespace(link_name=start_link, position=[0.0, -0.02, start_offset]),
        ]
        ends[hand] = [
            SimpleNamespace(link_name=end_link, position=[0.0, 0.02, -end_offset]),
            SimpleNamespace(link_name=end_link, position=[0.0, 0.02, end_offset]),
        ]

    robot = SimpleNamespace(
        links=links,
        assisted_grasp_start_points=starts,
        assisted_grasp_end_points=ends,
    )
    backend = RealCuroboBackend(None)
    backend._robot = robot

    left = backend.get_assisted_grasp_outward_ray_geometry("left")
    right = backend.get_assisted_grasp_outward_ray_geometry("right")

    assert left["outward_offset_m"] == pytest.approx(0.016)
    assert right["outward_offset_m"] == pytest.approx(0.012)
    assert left["start_outward_offset_m"] == pytest.approx(0.015)
    assert left["end_outward_offset_m"] == pytest.approx(0.016)
    assert left["outward_offset_selection"] == "positive_z_endpoint_max"
    assert left["ray_span_m"] == pytest.approx(np.hypot(0.04, 0.001))
    assert right["ray_span_m"] == pytest.approx(0.04)
    assert "finger_a" not in str(left)
    assert "finger_b" not in str(right)


def test_real_backend_ag_ray_geometry_missing_fails_closed():
    backend = RealCuroboBackend(None)
    backend._robot = SimpleNamespace(
        links={
            "left_eef_link": SimpleNamespace(
                get_position_orientation=lambda: (
                    np.zeros(3),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            )
        },
        assisted_grasp_start_points={"left": None},
        assisted_grasp_end_points={"left": None},
    )

    with pytest.raises(RuntimeError, match="rays are unavailable"):
        backend.get_assisted_grasp_outward_ray_geometry("left")


def test_gripper_command_stages_close_and_preserves_current_hold_segments():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    result = executor.release(hand="left", opening=0.0, retreat_m=0.0)

    assert result["primitive_success"] is True
    first = env.calls[0][0]
    commands = np.asarray(
        [call[0, ENV_ACTION_SEGMENTS["left_gripper"]][0] for call in env.calls]
    )
    assert commands[0] == pytest.approx(0.95)
    assert commands[-1] == pytest.approx(-1.0)
    assert np.max(np.abs(np.diff(commands))) <= 0.05 + 1e-6
    fine_deltas = np.abs(np.diff(commands))[commands[1:] < 0.0]
    assert np.max(fine_deltas) <= 0.00625 + 1e-6
    profile = result["metrics"]["gripper_command_profile"]
    assert profile["planned_steps"] == 180
    assert profile["coarse_max_command_step"] == pytest.approx(0.05)
    assert profile["fine_max_command_step"] == pytest.approx(0.00625)
    for segment_name in ("base", "trunk", "left_arm", "right_arm", "right_gripper"):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_allclose(first[segment], backend.hold[segment])


class _FakeTravMap:
    def __init__(self):
        self.floor_map = [np.zeros((3, 4), dtype=np.uint8)]
        self.floor_map[0][1, 2] = 255

    def world_to_map(self, _xy):
        return np.array([1, 2])


def test_traversability_uses_floor_map_row_column_and_fails_closed():
    backend = RealCuroboBackend(None)
    candidate = np.array([0.0, 0.0, 0.0])
    trav_map = _FakeTravMap()

    assert backend._candidate_is_traversable(trav_map, candidate, floor=0) is True
    trav_map.floor_map[0][1, 2] = 0
    assert backend._candidate_is_traversable(trav_map, candidate, floor=0) is False
    assert backend._candidate_is_traversable(object(), candidate, floor=0) is False


def test_cpu_joint_interpolation_bounds_every_waypoint_delta():
    result = _interpolate_joint_trajectory(
        np.array([[0.0, 0.0], [0.025, -0.011]], dtype=np.float32),
        max_inter_dist=0.01,
    )

    np.testing.assert_allclose(result[0], [0.0, 0.0])
    np.testing.assert_allclose(result[-1], [0.025, -0.011])
    assert np.max(np.abs(np.diff(result, axis=0))) <= 0.01 + 1e-6


def test_joint_retime_preserves_path_endpoints_and_command_dynamics_margin():
    trajectory = np.array(
        [[0.0, 0.0], [0.004, -0.002], [0.008, -0.004], [0.008, -0.004]],
        dtype=np.float32,
    )

    result, report = _retime_joint_trajectory(
        trajectory,
        sample_dt_s=1.0 / 60.0,
        max_command_velocity=3.0,
        max_command_acceleration=7.5,
    )

    np.testing.assert_allclose(result[0], trajectory[0])
    np.testing.assert_allclose(result[-1], trajectory[-1])
    assert len(result) > len(trajectory)
    assert report["path_geometry"] == "original_joint_polyline"
    assert report["full_collision_recheck_required"] is True
    assert report["max_command_velocity"] <= 3.0 + 1e-6
    assert report["max_command_acceleration"] <= 7.5 + 1e-6


def test_base_goal_matches_official_starter_virtual_joint_pose_semantics():
    class Robot:
        base_idx = np.arange(6)

        def get_joint_positions(self):
            return np.array([0.0, 0.0, 9.0, -0.7, 0.6, -1.2])

    backend = RealCuroboBackend(None)
    pos, quat = backend._base_target_world_pose(Robot(), np.array([1.0, 2.0, -0.35]))
    roll, pitch, yaw = _quat_to_intrinsic_rpy(quat)

    np.testing.assert_allclose(pos, [1.0, 2.0, 9.0])
    np.testing.assert_allclose([roll, pitch, yaw], [-0.7, 0.6, -0.35], atol=1e-7)


def test_base_workspace_limit_covers_scene_outside_og_fixed_five_metres():
    class SceneObject:
        def __init__(self, position):
            self.position = position

        def get_position_orientation(self):
            return np.asarray(self.position), np.array([0.0, 0.0, 0.0, 1.0])

    class Robot:
        scene = type(
            "Scene",
            (),
            {"objects": [SceneObject([8.4, -7.2, 0.0])]},
        )()

    backend = RealCuroboBackend(None)
    backend._base_xy_yaw = lambda _robot: np.array([5.213, 5.305, 0.0, 0.0])

    assert np.isclose(backend._base_prismatic_workspace_limit(Robot()), 10.4)


def test_base_candidates_share_one_official_world_collision_query():
    class Generator:
        def __init__(self):
            self.calls = []

        def check_collisions(self, q, **kwargs):
            self.calls.append((np.asarray(q), kwargs))
            return np.array([False, True, False])

    generator = Generator()
    backend = RealCuroboBackend(None)
    backend._generator = lambda **_kwargs: generator
    backend._initial_joint_pos_for_base_candidate = lambda _robot, candidate: np.array(
        [candidate[0], candidate[1], candidate[2]]
    )
    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.1]),
        np.array([3.0, 0.0, 0.2]),
    ]

    reports = backend._candidate_base_collision_reports(object(), candidates)

    assert len(generator.calls) == 1
    assert generator.calls[0][0].shape == (3, 3)
    assert generator.calls[0][1]["self_collision_check"] is False
    assert [report["colliding"] for report in reports] == [False, True, False]


def test_base_candidate_search_batches_collision_and_keeps_ranked_alternatives():
    class TravMap:
        def __init__(self):
            self.calls = 0

        def get_shortest_path(self, _floor, start, goal, **_kwargs):
            self.calls += 1
            return [np.asarray(start), np.asarray(goal)], float(
                np.linalg.norm(np.asarray(goal) - np.asarray(start))
            )

    trav_map = TravMap()
    backend = RealCuroboBackend(None)
    backend._scene = lambda _robot: type("Scene", (), {"trav_map": trav_map})()
    backend._record_base_phase = lambda _event: None
    backend._base_xy_yaw = lambda _robot: np.zeros(4)
    backend._current_floor = lambda _scene, _current: 0
    backend._candidate_is_traversable = lambda *_args, **_kwargs: True
    collision_batch_sizes = []

    def collision_reports(_robot, candidates):
        collision_batch_sizes.append(len(candidates))
        return [{"available": True, "colliding": False} for _candidate in candidates]

    backend._candidate_base_collision_reports = collision_reports
    backend.check_candidate_arm_reachability = lambda **_kwargs: (
        True,
        "reachable_candidate",
        {
            "reachability_stage": "candidate_world_collision_multi_orientation_ik",
            "selected_target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )

    ranked = backend._ranked_base_candidates(
        object(),
        hand="left",
        target_xyz=np.array([2.0, 0.0, 0.8]),
        standoff_m=0.85,
        deadline=time.monotonic() + 10.0,
    )

    assert len(ranked) == 6
    assert trav_map.calls == 9
    assert collision_batch_sizes == [9]
    assert backend._last_base_candidate_summary["candidate_batch_size"] == 9
    assert backend._last_base_candidate_summary["candidate_limit"] == 6


def test_base_candidate_collision_timeout_is_never_downgraded_to_unavailable():
    backend = RealCuroboBackend(None)
    backend._generator = lambda **_kwargs: type(
        "Generator",
        (),
        {
            "check_collisions": staticmethod(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    TimeoutError("injected collision-world timeout")
                )
            )
        },
    )()
    backend._initial_joint_pos_for_base_candidate = lambda _robot, _candidate: np.zeros(
        3
    )

    with pytest.raises(TimeoutError, match="collision-world timeout"):
        backend._candidate_base_collision_reports(object(), [np.array([1.0, 0.0, 0.0])])


def test_base_candidate_collision_world_refreshes_once_per_sim_step():
    calls = []

    class Generator:
        @staticmethod
        def check_collisions(q, **kwargs):
            calls.append(bool(kwargs["skip_obstacle_update"]))
            return np.zeros(len(q), dtype=bool)

    env = SimpleNamespace(_env_steps=4)
    backend = RealCuroboBackend(env)
    backend._generator = lambda **_kwargs: Generator()
    backend._initial_joint_pos_for_base_candidate = lambda _robot, _candidate: np.zeros(
        3
    )
    candidates = [np.array([1.0, 0.0, 0.0])]

    backend._candidate_base_collision_reports(object(), candidates)
    backend._candidate_base_collision_reports(object(), candidates)
    env._env_steps = 5
    backend._candidate_base_collision_reports(object(), candidates)

    assert calls == [False, True, False]


def test_base_reachability_checks_safe_precontact_point_toward_candidate():
    target = np.array([2.0, 1.0, 0.8])
    candidate = np.array([1.0, 1.0, 0.0])

    reachability_target = RealCuroboBackend._candidate_reachability_target(
        target,
        candidate,
    )

    np.testing.assert_allclose(reachability_target, [1.85, 1.0, 0.8])


def test_real_position_only_plan_preserves_current_eef_world_orientation(tmp_path):
    class Generator:
        batch_size = 2

        def __init__(self):
            self.target_quats = None

        def compute_trajectories(self, _positions, target_quats, **_kwargs):
            self.target_quats = np.asarray(target_quats)
            return np.array([False, False]), [None, None]

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand: tmp_path / "official_arm.yaml"
    current_quat = np.array([0.1, -0.2, 0.3, 0.9])
    current_quat /= np.linalg.norm(current_quat)
    backend.get_eef_pose = lambda _hand: (np.zeros(3), current_quat)

    result = backend._compute_arm_plan(
        hand="left",
        target_xyz=np.array([0.2, 0.1, 0.8]),
        target_quat_xyzw=None,
        timeout_s=1.0,
        ik_only=True,
    )

    assert result["ok"] is False
    assert result["metrics"]["orientation_mode"] == (
        "preserve_current_eef_world_orientation"
    )
    np.testing.assert_allclose(generator.target_quats, np.stack([current_quat] * 2))


def test_candidate_reachability_rotates_eef_orientation_with_candidate_base(tmp_path):
    class Generator:
        batch_size = 2

        def __init__(self):
            self.target_quats = None

        def compute_trajectories(self, _positions, target_quats, **_kwargs):
            self.target_quats = np.asarray(target_quats)
            return np.array([False, False]), [None, None]

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: object()
    backend._base_xy_yaw = lambda _robot: np.array([0.0, 0.0, 0.0, 0.0])
    backend._initial_joint_pos_for_base_candidate = lambda _robot, _candidate: np.zeros(
        28, dtype=np.float32
    )
    backend._hand_config_path = lambda _hand: tmp_path / "official_arm.yaml"
    backend.get_eef_pose = lambda _hand: (
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    result = backend._compute_arm_plan(
        hand="right",
        target_xyz=np.array([0.2, 0.1, 0.8]),
        target_quat_xyzw=None,
        timeout_s=1.0,
        ik_only=True,
        base_xyyaw=np.array([1.0, 0.0, np.pi / 2.0]),
    )

    expected = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    assert result["metrics"]["orientation_mode"] == (
        "preserve_eef_orientation_relative_to_candidate_base"
    )
    np.testing.assert_allclose(generator.target_quats, np.stack([expected] * 2))


def test_arm_curobo_call_has_hard_deadline_and_runtime_remains_usable(tmp_path):
    class Generator:
        batch_size = 2
        delay_s = 0.05

        def compute_trajectories(self, _positions, _target_quats, **_kwargs):
            time.sleep(self.delay_s)
            return np.array([False, False]), [None, None]

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand: tmp_path / "official_arm.yaml"
    backend.get_eef_pose = lambda _hand: (
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="ARM cuRobo"):
        backend._compute_arm_plan(
            hand="left",
            target_xyz=np.array([0.2, 0.1, 0.8]),
            target_quat_xyzw=None,
            timeout_s=0.01,
            ik_only=True,
        )
    assert time.monotonic() - started < 0.2

    generator.delay_s = 0.0
    result = backend._compute_arm_plan(
        hand="left",
        target_xyz=np.array([0.2, 0.1, 0.8]),
        target_quat_xyzw=None,
        timeout_s=0.1,
        ik_only=True,
    )
    assert result["ok"] is False


def test_wall_clock_deadline_fails_closed_off_main_thread():
    failures = []

    def call() -> None:
        try:
            with _wall_clock_deadline(0.1, "threaded planner"):
                pass
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    worker.join(timeout=1.0)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "main-thread dispatch" in str(failures[0])


def test_joint_target_conversion_calls_official_robot_q_to_action():
    class Controller:
        def __init__(self, command_dim):
            self.command_dim = command_dim

    class Robot:
        controllers = {
            "base": Controller(3),
            "trunk": Controller(4),
            "arm_left": Controller(7),
            "gripper_left": Controller(1),
            "arm_right": Controller(7),
            "gripper_right": Controller(1),
        }

        def __init__(self):
            self.calls = []

        def q_to_action(self, q):
            self.calls.append(q.clone())
            return q[:23]

    robot = Robot()
    backend = RealCuroboBackend(None)
    backend._robot = robot
    action = backend.joint_target_to_action(np.arange(23, dtype=np.float32), hand=None)

    assert len(robot.calls) == 1
    expected = np.arange(23, dtype=np.float32)
    expected[[14, 22]] = 1.0
    np.testing.assert_allclose(action, expected)
    np.testing.assert_allclose(action[[14, 22]], [1.0, 1.0])


def test_velocity_base_hold_uses_current_joint_targets_and_zero_base_velocity():
    class HolonomicBaseJointController:
        command_dim = 3
        motor_type = "velocity"
        dof_idx = np.array([0, 1, 2])

        @staticmethod
        def _reverse_preprocess_command(command):
            return command

    class JointController:
        use_delta_commands = False

        def __init__(self, indices):
            self.dof_idx = np.asarray(indices)
            self.command_dim = len(indices)

        @staticmethod
        def _reverse_preprocess_command(command):
            return command

    class MultiFingerGripperController:
        command_dim = 1

    class Robot:
        controllers = {
            "base": HolonomicBaseJointController(),
            "trunk": JointController(range(3, 7)),
            "arm_left": JointController(range(7, 14)),
            "gripper_left": MultiFingerGripperController(),
            "arm_right": JointController(range(14, 21)),
            "gripper_right": MultiFingerGripperController(),
        }

        @staticmethod
        def get_joint_positions():
            import torch

            return torch.arange(28, dtype=torch.float32)

    facade = SimpleNamespace(_gripper_latch={"left": -1.0, "right": 1.0})
    backend = RealCuroboBackend(facade)
    backend._robot = Robot()

    action = backend.velocity_base_hold_action()

    np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["base"]], 0.0)
    np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["trunk"]], np.arange(3, 7))
    np.testing.assert_allclose(
        action[ENV_ACTION_SEGMENTS["left_arm"]], np.arange(7, 14)
    )
    np.testing.assert_allclose(
        action[ENV_ACTION_SEGMENTS["right_arm"]], np.arange(14, 21)
    )
    assert action[ENV_ACTION_SEGMENTS["left_gripper"]][0] == -1.0
    assert action[ENV_ACTION_SEGMENTS["right_gripper"]][0] == 1.0


def test_fixed_world_base_q_is_reconverted_to_local_action_after_root_drift():
    class Controller:
        def __init__(self, command_dim):
            self.command_dim = command_dim

    class Robot:
        controllers = {
            "base": Controller(3),
            "trunk": Controller(4),
            "arm_left": Controller(7),
            "gripper_left": Controller(1),
            "arm_right": Controller(7),
            "gripper_right": Controller(1),
        }
        base_idx = list(range(6))
        trunk_control_idx = list(range(6, 10))
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }

        def __init__(self):
            self.q = np.zeros(28, dtype=np.float32)
            self.q[:6] = [1.0, 2.0, 0.4, 0.1, -0.2, 0.3]
            self.root_x = 0.0
            self.root_yaw = 0.0
            self.calls = []

        def get_joint_positions(self):
            return self.q.copy()

        def q_to_action(self, q):
            target = q.detach().cpu().numpy().copy()
            self.calls.append(target)
            action = np.zeros(23, dtype=np.float32)
            action[ENV_ACTION_SEGMENTS["base"]] = [
                target[0] - self.root_x,
                target[1],
                target[5] - self.root_yaw,
            ]
            return action

    robot = Robot()
    backend = RealCuroboBackend(None)
    backend._robot = robot
    reference = backend.capture_trajectory_hold_reference(hand="left")
    moving_target = np.full(28, 9.0, dtype=np.float32)

    first = backend.joint_target_to_action(
        moving_target, hand="left", fixed_reference=reference
    )
    robot.root_x = 0.25
    robot.root_yaw = 0.1
    second = backend.joint_target_to_action(
        moving_target, hand="left", fixed_reference=reference
    )

    np.testing.assert_allclose(robot.calls[0][:6], robot.q[:6])
    np.testing.assert_allclose(robot.calls[1][:6], robot.q[:6])
    np.testing.assert_allclose(first[ENV_ACTION_SEGMENTS["base"]], [1.0, 2.0, 0.3])
    np.testing.assert_allclose(second[ENV_ACTION_SEGMENTS["base"]], [0.75, 2.0, 0.2])
    assert (
        backend.locked_gripper_command_report(action=second, reference=reference)["ok"]
        is True
    )
    changed_command = second.copy()
    changed_command[ENV_ACTION_SEGMENTS["left_gripper"]] = 0.5
    assert (
        backend.locked_gripper_command_report(
            action=changed_command, reference=reference
        )["ok"]
        is False
    )

    base_reference = backend.capture_trajectory_hold_reference(hand=None)
    backend.joint_target_to_action(
        moving_target, hand=None, fixed_reference=base_reference
    )
    np.testing.assert_allclose(robot.calls[-1][:6], moving_target[:6])
    np.testing.assert_allclose(robot.calls[-1][6:24], robot.q[6:24])


def test_base_waypoint_tracking_is_tighter_than_final_arrival_tolerance():
    class Robot:
        base_control_idx = [0, 1, 5]
        trunk_control_idx = [6, 7, 8, 9]
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }

        @staticmethod
        def get_joint_positions():
            return np.zeros(28, dtype=np.float32)

        @staticmethod
        def get_joint_velocities():
            return np.zeros(28, dtype=np.float32)

    backend = RealCuroboBackend(None)
    backend._robot = Robot()
    target = np.zeros(28, dtype=np.float32)
    target[0] = 0.02

    report = backend.joint_tracking_report(target, hand=None)

    assert report["reached"] is False
    assert report["base_waypoint_xy_tolerance_m"] == 0.01
    target[0] = 0.005
    assert backend.joint_tracking_report(target, hand=None)["reached"] is True


def test_arm_waypoint_tracking_tolerates_small_controller_settling_error():
    class Robot:
        base_control_idx = [0, 1, 5]
        trunk_control_idx = [6, 7, 8, 9]
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }

        @staticmethod
        def get_joint_positions():
            return np.zeros(28, dtype=np.float32)

    backend = RealCuroboBackend(None)
    backend._robot = Robot()
    target = np.zeros(28, dtype=np.float32)
    target[10] = 0.011

    report = backend.joint_tracking_report(target, hand="left")

    assert report["reached"] is True
    assert report["articulation_waypoint_tolerance_rad"] == 0.015
    target[10] = 0.016
    assert backend.joint_tracking_report(target, hand="left")["reached"] is False


def test_arm_waypoint_tracking_ignores_locked_non_active_arm_error():
    class Robot:
        base_control_idx = [0, 1, 5]
        trunk_control_idx = [6, 7, 8, 9]
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }

        @staticmethod
        def get_joint_positions():
            current = np.zeros(28, dtype=np.float32)
            current[17] = 0.02
            return current

    backend = RealCuroboBackend(None)
    backend._robot = Robot()
    target = np.zeros(28, dtype=np.float32)

    left_report = backend.joint_tracking_report(target, hand="left")
    right_report = backend.joint_tracking_report(target, hand="right")

    assert left_report["reached"] is True
    assert left_report["max_articulation_error_rad"] == 0.0
    assert right_report["reached"] is False
    assert right_report["max_articulation_error_rad"] == pytest.approx(0.02)


def test_real_locked_joint_reference_uses_units_wrap_and_full_base_dofs():
    class Robot:
        base_idx = list(range(6))
        trunk_control_idx = [6, 7, 8, 9]
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }
        gripper_control_idx = {"left": [24, 25], "right": [26, 27]}

        def __init__(self):
            self.q = np.zeros(28, dtype=np.float64)

        def get_joint_positions(self):
            return self.q.copy()

    robot = Robot()
    robot.q[5] = np.pi - 0.005
    backend = RealCuroboBackend(None)
    backend._robot = robot

    right_arm_reference = backend.capture_locked_joint_reference(hand="right")
    robot.q[5] = -np.pi + 0.005
    wrapped_report = backend.locked_joint_drift_report(reference=right_arm_reference)
    assert wrapped_report["ok"] is True
    assert wrapped_report["base_rpy_drift_rad"] == pytest.approx(0.01)

    robot.q[2] = 0.011
    base_z_report = backend.locked_joint_drift_report(reference=right_arm_reference)
    assert base_z_report["ok"] is False
    assert base_z_report["base_z_drift_m"] == pytest.approx(0.011)

    robot.q[2] = 0.0
    robot.q[10] = 0.01616
    right_arm_report = backend.locked_joint_drift_report(reference=right_arm_reference)

    assert right_arm_report["ok"] is False
    assert right_arm_report["articulation_drift_rad"] == pytest.approx(0.01616)

    # Compliant physical gripper joints are deliberately outside the locked
    # q-drift scope; only their fixed packed controller commands are checked.
    robot.q[10] = 0.0
    robot.q[24:28] = 0.5
    compliant_gripper_report = backend.locked_joint_drift_report(
        reference=right_arm_reference
    )
    assert compliant_gripper_report["ok"] is True

    robot.q[:] = 0.0
    base_reference = backend.capture_locked_joint_reference(hand=None)
    robot.q[0] = 0.2  # active base motion is not part of the BASE lock set
    base_report = backend.locked_joint_drift_report(reference=base_reference)
    assert base_report["ok"] is True
    assert base_report["base_xy_drift_m"] is None
    assert base_report["articulation_drift_rad"] == 0.0


def test_joint_margin_uses_raw_or_three_percent_range_per_joint():
    class Robot:
        trunk_control_idx = list(range(4))
        arm_control_idx = {
            "left": list(range(4, 11)),
            "right": list(range(11, 18)),
        }
        dof_names_ordered = [f"joint_{index}" for index in range(18)]
        lower = np.full(18, -1.0, dtype=np.float32)
        upper = np.full(18, 1.0, dtype=np.float32)
        q = np.zeros(18, dtype=np.float32)
        control_limits = {"position": (lower, upper)}

        @classmethod
        def get_joint_positions(cls, normalized=False):
            if normalized:
                return 2.0 * (cls.q - cls.lower) / (cls.upper - cls.lower) - 1.0
            return cls.q.copy()

    backend = RealCuroboBackend(None)
    backend._robot = Robot()

    Robot.q[-1] = 0.96
    unsafe = backend.joint_margin_report()
    assert unsafe["ok"] is False
    assert unsafe["min_raw_margin_joint_units"] == pytest.approx(0.04)
    assert unsafe["min_range_fraction"] == pytest.approx(0.02)
    assert unsafe["limiting_joint"] == "joint_17"

    Robot.lower[-1] = 0.0
    Robot.upper[-1] = 1.0
    Robot.q[-1] = 0.96
    range_safe = backend.joint_margin_report()
    assert range_safe["ok"] is True
    assert range_safe["min_raw_margin_joint_units"] == pytest.approx(0.04)
    assert range_safe["min_range_fraction"] == pytest.approx(0.04)


def test_planner_warmup_delegates_to_backend_and_requires_implementation(tmp_path):
    class Backend:
        def warmup(self):
            return {"status": "complete", "elapsed_s": 1.25}

        def warmup_prepress(self, *, hand, expected_attached_root):
            return {
                "status": "complete",
                "generator_kind": "prepress_arm",
                "held_hand": hand,
                "expected_attached_root": expected_attached_root,
            }

    planner = PlannerExecutor(
        env=object(),
        frame_cache=FrameCache(),
        output_dir=tmp_path,
        backend=Backend(),
    )
    assert planner.warmup() == {"status": "complete", "elapsed_s": 1.25}
    expected_root = object()
    assert planner.warmup_prepress(
        hand="left", expected_attached_root=expected_root
    ) == {
        "status": "complete",
        "generator_kind": "prepress_arm",
        "held_hand": "left",
        "expected_attached_root": expected_root,
    }

    planner.backend = object()
    with pytest.raises(RuntimeError, match="safety warmup"):
        planner.warmup()
    with pytest.raises(RuntimeError, match="pre-press safety warmup"):
        planner.warmup_prepress(hand="left", expected_attached_root=expected_root)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_prepress_warmup_is_attachment_aware_and_phase_scoped(tmp_path, hand):
    calls = []
    q = np.linspace(-0.2, 0.2, 24, dtype=np.float32)

    class Robot:
        @staticmethod
        def get_joint_positions():
            return q.copy()

    class Root:
        prim_path = "/World/radio/root"

    expected_root = Root()
    attached = {f"{hand}_eef": expected_root}
    generator = object()
    backend = RealCuroboBackend.__new__(RealCuroboBackend)
    backend.output_dir = tmp_path
    backend._find_robot = lambda: Robot()
    backend.get_attached_object = lambda selected: (
        calls.append(("attachment", selected)) or attached
    )
    backend._generator = lambda *, kind, hand: (
        calls.append(("generator", kind, hand)) or generator
    )
    backend._check_q_trajectory_collisions = (
        lambda selected_generator, selected_q, *, attached_obj, skip_obstacle_update: (
            calls.append(
                (
                    "collision",
                    selected_generator,
                    np.asarray(selected_q).copy(),
                    attached_obj,
                    skip_obstacle_update,
                )
            )
            or {"available": True, "colliding": False, "checked_waypoints": 1}
        )
    )
    backend.get_eef_pose = lambda selected: (
        calls.append(("eef", selected))
        or (np.array([0.4, 0.5, 0.6]), np.array([0.0, 0.0, 0.0, 1.0]))
    )
    backend._compute_arm_plan = lambda **kwargs: (
        calls.append(("plan", kwargs))
        or {"ok": True, "joint_trajectory": q.reshape(1, -1), "metrics": {}}
    )

    report = backend.warmup_prepress(hand=hand, expected_attached_root=expected_root)

    assert report["status"] == "complete"
    assert report["generator_kind"] == "prepress_arm"
    assert report["held_hand"] == hand
    assert report["base_generator_warmed"] is False
    assert report["unrelated_press_arm_generator_warmed"] is False
    assert report["robot_q_pose_jump_max"] == pytest.approx(0.0)
    assert calls[0] == ("attachment", hand)
    assert calls[1] == ("generator", "prepress_arm", hand)
    collision = next(item for item in calls if item[0] == "collision")
    assert collision[1] is generator
    assert collision[3] is attached
    assert collision[4] is False
    connected_collision = [item for item in calls if item[0] == "collision"][1]
    assert connected_collision[2].shape[0] == 2
    np.testing.assert_allclose(connected_collision[2][0], q)
    np.testing.assert_allclose(connected_collision[2][1], q)
    assert connected_collision[3] is attached
    plan = next(item[1] for item in calls if item[0] == "plan")
    assert plan["hand"] == hand
    assert plan["attached_obj"] is attached
    assert plan["generator_kind"] == "prepress_arm"
    assert plan["ik_only"] is False
    assert (tmp_path / "planner_prepress_warmup.json").is_file()


def test_prepress_warmup_fails_closed_on_current_attached_collision(tmp_path):
    class Robot:
        @staticmethod
        def get_joint_positions():
            return np.zeros(24, dtype=np.float32)

    backend = RealCuroboBackend.__new__(RealCuroboBackend)
    backend.output_dir = tmp_path
    backend._find_robot = lambda: Robot()
    expected_root = object()
    backend.get_attached_object = lambda _hand: {"eef": expected_root}
    backend._generator = lambda **_kwargs: object()
    backend._check_q_trajectory_collisions = lambda *_args, **_kwargs: {
        "available": True,
        "colliding": True,
    }
    backend._compute_arm_plan = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("planning must not run from a colliding current state")
    )

    with pytest.raises(RuntimeError, match="attached collision check failed"):
        backend.warmup_prepress(hand="right", expected_attached_root=expected_root)

    artifact = json.loads(
        (tmp_path / "planner_prepress_warmup.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "error"
    assert artifact["stages"]["current_q_attached_combined_collision"]["ok"] is False


def test_q_trajectory_repeats_each_waypoint_until_closed_loop_reached():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.action_calls = 0
            self.tracking_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            self.action_calls += 1
            return np.zeros(23, dtype=np.float32)

        def joint_tracking_report(self, target_q, *, hand):
            self.tracking_calls += 1
            reached = self.tracking_calls % 3 == 0
            return {
                "available": True,
                "reached": reached,
                "max_articulation_error_rad": 0.0 if reached else 0.1,
                "max_base_xy_error_m": 0.0,
                "base_yaw_error_rad": 0.0,
            }

    backend = Backend()
    executor, _ = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        joint_trajectory=np.zeros((2, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    assert backend.action_calls == 6
    assert result["diagnostics"]["trace_steps_persisted"] == 6


def test_arm_q_trajectory_uses_one_fixed_hold_for_inactive_segments():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.fixed_hold = np.arange(23, dtype=np.float32) * 0.01

        def hold_action(self, hand=None):
            assert hand == "right"
            return self.fixed_hold.copy()

        def joint_target_to_action(self, target_q, *, hand):
            del target_q
            assert hand == "right"
            action = np.full(23, 9.0, dtype=np.float32)
            action[ENV_ACTION_SEGMENTS["trunk"]] = 2.0
            action[ENV_ACTION_SEGMENTS["right_arm"]] = 3.0
            return action

    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="right",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        joint_trajectory=np.zeros((1, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    action = env.calls[0][0]
    for segment_name in ("base", "left_arm", "left_gripper", "right_gripper"):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_allclose(action[segment], backend.fixed_hold[segment])
    np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["trunk"]], 2.0)
    np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["right_arm"]], 3.0)


def test_base_q_trajectory_locks_trunk_both_arms_and_grippers():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.fixed_hold = np.arange(23, dtype=np.float32) * 0.02

        def hold_action(self, hand=None):
            assert hand is None
            return self.fixed_hold.copy()

        def joint_target_to_action(self, target_q, *, hand):
            del target_q
            assert hand is None
            action = np.full(23, 8.0, dtype=np.float32)
            action[ENV_ACTION_SEGMENTS["base"]] = [0.1, 0.2, 0.3]
            return action

    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        base_goal_xyyaw=np.zeros(3, dtype=np.float64),
        hold_steps_required=1,
        joint_trajectory=np.zeros((1, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    action = env.calls[0][0]
    np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["base"]], [0.1, 0.2, 0.3])
    for segment_name in (
        "trunk",
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
    ):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_allclose(action[segment], backend.fixed_hold[segment])


def test_q_trajectory_inactive_joint_drift_is_a_bounded_failure():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.drift_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            del target_q, hand
            return np.zeros(23, dtype=np.float32)

        def locked_joint_drift_report(self, *, reference):
            del reference
            self.drift_calls += 1
            drift = 0.0 if self.drift_calls == 1 else 0.01616
            return {
                "available": True,
                "ok": drift <= 0.01,
                "base_xy_drift_m": 0.0,
                "base_xy_threshold_m": 0.01,
                "base_z_drift_m": 0.0,
                "base_z_threshold_m": 0.01,
                "base_rpy_drift_rad": 0.0,
                "base_rpy_threshold_rad": np.deg2rad(1.0),
                "articulation_drift_rad": drift,
                "articulation_threshold_rad": 0.01,
                "locked_joint_count": 10,
            }

    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="right",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        joint_trajectory=np.zeros((3, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "locked_joint_drift"
    assert result["metrics"]["locked_joint_peaks"][
        "articulation_drift_rad"
    ] == pytest.approx(0.01616)
    assert len(env.calls) == 2


def test_post_step_joint_margin_cannot_be_bypassed_by_success():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.margin_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            del target_q, hand
            return np.zeros(23, dtype=np.float32)

        def joint_margin_report(self):
            self.margin_calls += 1
            safe = self.margin_calls == 1
            return {
                "available": True,
                "ok": safe,
                "min_raw_margin_joint_units": 0.06 if safe else 0.01,
                "min_range_fraction": 0.04 if safe else 0.01,
                "threshold_raw_rad": 0.05,
                "threshold_range_fraction": 0.03,
            }

    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        joint_trajectory=np.zeros((1, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "joint_limit_margin"
    assert len(env.calls) == 1
    assert result["metrics"]["joint_margin"]["ok"] is False


def test_every_q_waypoint_forces_fresh_runtime_collision_query():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.collision_forces = []

        def joint_target_to_action(self, target_q, *, hand):
            return np.zeros(23, dtype=np.float32)

        def collision_report(self, *, force=False):
            self.collision_forces.append(force)
            return {
                "available": True,
                "colliding": False,
                "min_margin_m": 0.005,
                "margin_available": True,
            }

    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        runtime_collision_interval_steps=3,
        joint_trajectory=np.zeros((4, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    assert len(env.calls) == 4
    assert backend.collision_forces.count(True) == len(env.calls)
    assert backend.collision_forces[-1] is False  # final result reads cached proof


def test_guarded_terminal_actual_acceleration_is_a_bounded_oscillation_failure():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__(contact_mode="expected_near_target")
            self.dynamics_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            del target_q, hand
            return np.zeros(23, dtype=np.float32)

        def dynamics_report(self):
            self.dynamics_calls += 1
            acceleration = 0.0 if self.dynamics_calls == 1 else 24.423
            return {
                "available": True,
                "ok": acceleration <= 15.0,
                "max_actual_velocity": 0.212,
                "max_actual_velocity_joint": "right_arm_joint5",
                "max_velocity_ratio": 0.1,
                "max_actual_acceleration": acceleration,
                "max_actual_acceleration_joint": "right_arm_joint5",
                "max_acceleration_limit": 15.0,
                "max_acceleration_ratio": acceleration / 15.0,
                "source": "feedback",
            }

    backend = Backend()
    executor, _ = _executor(backend)
    result = executor._execute_actions(
        None,
        hand="right",
        target_xyz=np.array([0.5, 0.0, 0.0]),
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=True,
        hold_steps_required=10,
        contact_target_xyz=np.array([0.5, 0.0, 0.0]),
        allow_expected_contact=True,
        joint_trajectory=np.zeros((1, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "guarded_terminal_oscillation"
    assert result["metrics"]["actual_acceleration_limit_rad_s2"] == 15.0
    assert result["metrics"]["terminal_hold_actual_dynamics_violation"] is True


def test_pick_and_release_require_assisted_grasp_state_confirmation():
    missing, _ = _executor(_FakeBackend(contact_mode="expected_near_target"))
    failed_pick = missing.pick(
        hand="left",
        target_xyz=[0.5, 0.0, -0.004],
        pregrasp_offset_m=0.004,
        lift_m=0.0,
    )
    assert failed_pick["stop_reason"] == "grasp_not_confirmed"

    held_backend = _FakeBackend(
        contact_mode="expected_near_target",
        attach_on_close=True,
    )
    held, _ = _executor(held_backend)
    successful_pick = held.pick(
        hand="left",
        target_xyz=[0.5, 0.0, -0.004],
        pregrasp_offset_m=0.004,
        lift_m=0.0,
    )
    assert successful_pick["primitive_success"] is True
    assert successful_pick["metrics"]["attachment_endpoint_held_steps"] == 10
    released = held.release(hand="left", retreat_m=0.0)
    assert released["primitive_success"] is True
    assert held_backend.get_attached_object("left") is None


def test_pick_fails_closed_when_assisted_grasp_ray_geometry_is_missing():
    backend = _FakeBackend(assisted_grasp_ray_offset_m=None)
    executor, env = _executor(backend)

    result = executor.pick(
        hand="right",
        target_xyz=[0.5, 0.0, -0.04],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "assisted_grasp_ray_geometry_unavailable"
    assert env.calls == []


def test_gripper_close_is_staged_and_holds_two_finger_contact_before_latch():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__(contact_mode="expected_near_target")
            self.advance_calls = 0

        def contact_report(self, **kwargs):
            report = super().contact_report(**kwargs)
            report.update(
                {
                    "target_finger_contact_count": 2,
                    "target_two_finger_contact": True,
                    "assisted_grasp_raycast_available": True,
                    "assisted_grasp_raycast_target_match": True,
                    "official_assisted_grasp_counter": min(self.advance_calls, 2),
                }
            )
            return report

        def advance(self):
            self.advance_calls += 1
            super().advance()
            if self.advance_calls >= 8 and self.attached_obj is None:
                self.attached_obj = {"left_eef_link": self.target_root}

    backend = Backend()
    executor, env = _executor(backend)
    expected = backend.resolve_target_attachment(
        hand="left", target_xyz=np.array([0.5, 0.0, 0.0])
    )

    result = executor._gripper_command(
        "left",
        opening=0.0,
        timeout_s=5.0,
        contact_target_xyz=np.array([0.5, 0.0, 0.0]),
        allow_expected_contact=True,
        hold_steps_required=10,
        stop_on_attachment=True,
        expected_attachment=expected,
    )

    assert result["primitive_success"] is True
    commands = np.asarray(
        [call[0, ENV_ACTION_SEGMENTS["left_gripper"]][0] for call in env.calls]
    )
    np.testing.assert_allclose(commands[:11], commands[0])
    assert np.max(np.abs(np.diff(commands))) <= 0.05 + 1e-6
    assert result["metrics"]["gripper_contact_settle_started"] is True
    assert result["metrics"]["gripper_contact_settle_steps_executed"] == 10
    assert result["metrics"]["attachment_confirmation_steps"] == 10
    assert result["metrics"]["gripper_command_profile"]["planned_steps"] == 180
    assert result["metrics"]["gripper_command_profile"][
        "max_command_step"
    ] == pytest.approx(0.05)
    assert result["metrics"]["gripper_command_profile"][
        "fine_max_command_step"
    ] == pytest.approx(0.00625)
    trace = result["diagnostics"]["trace"]
    assert trace
    assert all(isinstance(sample["gripper_command"], float) for sample in trace)


def test_gripper_close_rejects_wrong_attachment_root_immediately():
    class Backend(_FakeBackend):
        def advance(self):
            super().advance()
            self.attached_obj = {"left_eef_link": object()}

    backend = Backend()
    executor, env = _executor(backend)
    expected = backend.resolve_target_attachment(
        hand="left", target_xyz=np.array([0.5, 0.0, 0.0])
    )

    result = executor._gripper_command(
        "left",
        opening=0.0,
        timeout_s=5.0,
        hold_steps_required=10,
        stop_on_attachment=True,
        expected_attachment=expected,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_identity_mismatch"
    assert len(env.calls) == 1


def test_lift_checks_exact_attachment_each_step_and_fails_on_loss():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.advance_calls = 0
            self.attached_obj = {"left_eef_link": self.target_root}

        def advance(self):
            self.advance_calls += 1
            super().advance()
            if self.advance_calls == 3:
                self.attached_obj = None

    backend = Backend()
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0
    expected = backend.resolve_target_attachment(
        hand="left", target_xyz=np.array([0.5, 0.0, 0.0])
    )

    result = executor._execute_actions(
        np.zeros((5, 23), dtype=np.float32),
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=10,
        expected_attachment=expected,
        require_attachment=True,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_lost"
    assert len(env.calls) == 3


def test_pick_default_pregrasp_uses_full_eight_centimeter_offset():
    backend = _FakeBackend(
        contact_mode="expected_near_target",
        attach_on_close=True,
    )
    executor, _ = _executor(backend)
    target = np.array([0.5, 0.0, -0.1])

    result = executor.pick(
        hand="left",
        target_xyz=target,
        approach_vector=[0.0, 0.0, -1.0],
        grasp_quat_xyzw=[1.0, 0.0, 0.0, 0.0],
        lift_m=0.0,
    )

    assert result["primitive_success"] is True
    np.testing.assert_allclose(
        backend.planned_targets[0],
        target + np.array([0.0, 0.0, 0.119]),
    )


def test_pick_pregrasp_uses_shared_public_deadline_without_fixed_stage_cap(
    monkeypatch,
):
    backend = _FakeBackend()
    executor, _ = _executor(backend)
    seen = {}

    def fail_after_recording_timeout(**kwargs):
        seen["timeout_s"] = kwargs["timeout_s"]
        return {
            "primitive_success": False,
            "stop_reason": "timeout",
            "recoverable": True,
            "suggested_next_tool": "move_to",
            "metrics": {},
            "diagnostics": {},
        }

    monkeypatch.setattr(
        executor,
        "_move_to_composite_stage",
        fail_after_recording_timeout,
    )

    result = executor.pick(
        hand="right",
        target_xyz=[0.5, 0.0, -0.1],
        approach_vector=[0.0, 0.0, -1.0],
        grasp_quat_xyzw=[1.0, 0.0, 0.0, 0.0],
        timeout_s=120.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "timeout"
    assert 119.0 < seen["timeout_s"] <= 120.0


def test_pick_aligns_ag_ray_plane_instead_of_fingertip_to_rgbd_surface():
    class FingertipBackend(_FakeBackend):
        @staticmethod
        def get_eef_to_fingertip_length(hand):
            assert hand == "left"
            return 0.04

        def get_assisted_grasp_outward_ray_geometry(self, hand):
            geometry = super().get_assisted_grasp_outward_ray_geometry(hand)
            geometry["start_outward_offset_m"] = 0.015
            geometry["end_outward_offset_m"] = 0.016
            geometry["outward_offset_m"] = 0.016
            geometry["plane_mismatch_m"] = 0.001
            geometry["outward_offset_selection"] = "positive_z_endpoint_max"
            return geometry

    backend = FingertipBackend(
        attach_on_close=True,
        assisted_grasp_ray_offset_m=0.016,
    )
    executor, _ = _executor(backend)
    target = np.array([0.5, 0.0, -0.1])

    result = executor.pick(
        hand="left",
        target_xyz=target,
        approach_vector=[0.0, 0.0, -1.0],
        grasp_quat_xyzw=[1.0, 0.0, 0.0, 0.0],
        lift_m=0.0,
    )

    assert result["primitive_success"] is True
    np.testing.assert_allclose(
        backend.planned_targets[0],
        target + np.array([0.0, 0.0, 0.095]),
    )
    assert any(
        np.allclose(planned, target + np.array([0.0, 0.0, 0.015]))
        for planned in backend.planned_targets[1:]
    )
    geometry = result["metrics"]["pick_contact_geometry"]
    assert geometry["eef_to_fingertip_offset_m"] == pytest.approx(0.04)
    assert geometry["assisted_grasp_outward_ray_offset_m"] == pytest.approx(0.016)
    assert geometry["outward_ray_to_fingertip_delta_m"] == pytest.approx(-0.024)
    assert geometry["guarded_overtravel_m"] == pytest.approx(0.001)
    assert geometry["maximum_ray_endpoint_penetration_m"] == pytest.approx(0.001)
    assert geometry["assisted_grasp_ray_endpoint_penetration_m"] == {
        "start": pytest.approx(0.0),
        "end": pytest.approx(0.001),
    }
    np.testing.assert_allclose(geometry["target_xyz"], target)


def test_pick_reuses_guarded_path_only_after_attached_body_recheck():
    class CertifiedRetreatBackend(_FakeBackend):
        def __init__(self):
            super().__init__(
                contact_mode="expected_near_target",
                attach_on_close=True,
            )
            self.certifications = []

        def plan_guarded_ik_path(
            self,
            *,
            hand,
            target_xyz,
            target_quat_xyzw,
            timeout_s,
            attached_obj=None,
            contact_target_xyz=None,
        ):
            plan = self.plan_arm_trajectory(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                attached_obj=attached_obj,
            )
            contact = np.asarray(target_xyz, dtype=np.float32)
            lifted = contact + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            plan["reverse_joint_trajectory"] = np.stack(
                [np.r_[contact, np.zeros(25)], np.r_[lifted, np.zeros(25)]]
            ).astype(np.float32)
            return plan

        def certify_attached_joint_trajectory(
            self, *, hand, joint_trajectory, attached_obj, timeout_s
        ):
            self.certifications.append(
                {
                    "hand": hand,
                    "joint_trajectory": np.asarray(joint_trajectory).copy(),
                    "attached_obj": attached_obj,
                }
            )
            return {
                "ok": True,
                "stop_reason": "certified",
                "metrics": {"attached_collision_body": {"available": True}},
            }

        def joint_target_to_action(self, target_q, *, hand):
            self.target = np.asarray(target_q, dtype=np.float64)[:3]
            return np.zeros(23, dtype=np.float32)

    backend = CertifiedRetreatBackend()
    executor, _ = _executor(backend)
    target = np.array([0.5, 0.0, -0.08])

    result = executor.pick(
        hand="left",
        target_xyz=target,
        approach_vector=[0.0, 0.0, -1.0],
        grasp_quat_xyzw=[1.0, 0.0, 0.0, 0.0],
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "picked"
    assert len(backend.certifications) == 1
    assert backend.certifications[0]["attached_obj"] is not None
    assert result["metrics"]["lift_execution"] == {
        "method": "reverse_guarded_path",
        "full_attached_path_rechecked": True,
        "runtime_collision_interval_steps": 1,
        "physical_contact_query_interval_steps": 1,
        "certification": {"attached_collision_body": {"available": True}},
    }


def test_uniform_envelope_finishes_only_on_official_success():
    executor, env = _executor(_FakeBackend())
    env._last_info = {"done": {"success": True}}

    result = executor.pixel_to_world(camera="head", frame_id="f0", u=2, v=2)

    assert result["task_success"] is True
    assert result["_finish"] is True
    for key in (
        "position_error_m",
        "orientation_error_rad",
        "joint_margin",
        "collision_margin_m",
        "elapsed_s",
        "trace",
        "trace_artifact",
    ):
        assert key in result
