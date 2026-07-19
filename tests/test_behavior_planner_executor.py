import numpy as np

from robots.behavior.camera_geometry import CameraIntrinsics, FrameCache
from robots.behavior.planner_executor import (
    PlannerExecutor,
    RealCuroboBackend,
    _interpolate_joint_trajectory,
)
from robots.behavior.schemas import ENV_ACTION_SEGMENTS
from rpent.tools.toolkit import ToolResult


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
    ):
        self.progress = progress
        self.bad_actions = bad_actions
        self.bad_velocity = bad_velocity
        self.reachable = reachable
        self.contact_mode = contact_mode
        self.attached_obj = attached_obj
        self.pose = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.base_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target = None
        self.target_quat = None
        self.planned_targets = []
        self.attached_used = []
        self.hold = (np.arange(23, dtype=np.float32) + 1.0) * 0.01

    def check_arm_reachability(self, *, hand, target_xyz, target_quat_xyzw, base_xyyaw=None):
        if not self.reachable:
            return False, "navigation_required", {"eef_target_distance_m": 2.0}
        return True, "reachable_candidate", {
            "eef_target_distance_m": float(np.linalg.norm(self.pose - target_xyz)),
            "reachability_stage": "world_collision_ik" if base_xyyaw is None else "candidate_kinematic_ik",
        }

    def plan_arm_trajectory(self, *, hand, target_xyz, target_quat_xyzw, timeout_s, attached_obj=None):
        self.target = np.asarray(target_xyz, dtype=np.float64)
        self.target_quat = (
            None if target_quat_xyzw is None else np.asarray(target_quat_xyzw, dtype=np.float64)
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

    def contact_report(self, *, hand, target_xyz=None, allowed_contact_distance_m=0.025):
        if self.contact_mode == "unexpected":
            return {
                "available": True,
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
                "unexpected_contact": False,
                "expected_contact": True,
                "allowed_contact_distance_m": allowed_contact_distance_m,
            }
        return {
            "available": True,
            "unexpected_contact": False,
            "expected_contact": False,
            "allowed_contact_distance_m": allowed_contact_distance_m,
        }

    def get_attached_object(self, hand):
        return self.attached_obj


def _executor(backend):
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
    return PlannerExecutor(env=env, frame_cache=cache, backend=backend), env


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

        def _ranked_base_candidates(self, robot, *, hand, target_xyz, standoff_m):
            return [
                {
                    "xyyaw": np.array([1.0, 0.0, 0.0]),
                    "geodesic_distance_m": 1.0,
                    "reachability_reason": "reachable_candidate",
                    "reachability_stage": "candidate_kinematic_ik",
                },
                {
                    "xyyaw": np.array([0.0, 1.0, 1.57]),
                    "geodesic_distance_m": 1.2,
                    "reachability_reason": "reachable_candidate",
                    "reachability_stage": "candidate_kinematic_ik",
                },
            ]

        def _compute_base_plan(self, *, target_xyyaw, timeout_s):
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
    assert [attempt["ok"] for attempt in result["metrics"]["base_plan_attempts"]] == [False, True]


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
    ok = executor.rotate_wrist(hand="left", relative_axis_angle=[0.0, 0.0, 1.0, np.pi / 8])

    assert both["primitive_success"] is False
    assert "exactly one" in both["diagnostics"]["error"]
    assert bad["primitive_success"] is False
    assert "axis_x" in bad["diagnostics"]["error"]
    assert ok["primitive_success"] is True


def test_press_uses_guarded_two_millimeter_waypoints():
    backend = _FakeBackend()
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.01],
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


def test_guarded_contact_aborts_before_target_neighborhood():
    backend = _FakeBackend(contact_mode="expected")
    executor, env = _executor(backend)

    result = executor.pick(hand="left", target_xyz=[0.5, 0.0, -0.04], pregrasp_offset_m=0.08)

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


def test_rotate_wrist_passes_attached_body_to_planner_without_exposing_pose():
    attached = {"left_eef_link": object()}
    backend = _FakeBackend(attached_obj=attached)
    executor, _ = _executor(backend)

    result = executor.rotate_wrist(hand="left", relative_axis_angle=[0.0, 0.0, 1.0, 0.1])

    assert result["primitive_success"] is True
    assert attached in backend.attached_used
    assert "object at" not in str(result["diagnostics"])


def test_real_backend_wraps_assisted_grasp_object_by_selected_eef_link():
    attached = object()

    class Robot:
        _ag_obj_in_hand = {"left": attached, "right": None}

    backend = RealCuroboBackend(None)
    backend._robot = Robot()

    assert backend.get_attached_object("left") == {"left_eef_link": attached}
    assert backend.get_attached_object("right") is None


def test_gripper_command_preserves_current_hold_segments():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    result = executor.release(hand="left", opening=0.0, retreat_m=0.0)

    assert result["primitive_success"] is True
    first = env.calls[0][0]
    assert first[ENV_ACTION_SEGMENTS["left_gripper"]][0] == -1.0
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
