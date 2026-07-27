import hashlib
import importlib.util
import inspect
import json
import math
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import CameraIntrinsics, FrameCache
from robots.behavior.planner_executor import (
    BASE_ACTIVE_JOINT_NAMES,
    LOCAL_GUARDED_IK_SEEDS,
    PlannerExecutor,
    RealCuroboBackend,
    _apply_single_arm_isolation_mask,
    _attachment_identity_status,
    _guarded_waypoint_distances,
    _interpolate_joint_trajectory,
    _quat_to_intrinsic_rpy,
    _retime_joint_trajectory,
    _terminally_smoothed_joint_trajectory,
    _wall_clock_deadline,
)
from robots.behavior.schemas import ENV_ACTION_SEGMENTS, PI0_NAV_PICK_SPEC
from robots.behavior.toolkit import BehaviorToolResult

_REQUIRES_TORCH = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires the production torch tensor runtime",
)

_REMOVED_RUNTIME_REPORTER_CASES = (
    (
        "collision_report",
        {
            "available": True,
            "colliding": True,
            "min_margin_m": -0.001,
            "margin_available": True,
        },
    ),
    (
        "collision_report",
        {
            "available": False,
            "reason": "collision_feedback_unavailable",
            "colliding": False,
            "min_margin_m": None,
            "margin_available": False,
        },
    ),
    (
        "joint_margin_report",
        {
            "available": True,
            "ok": False,
            "min_raw_margin_joint_units": 0.01,
            "threshold_raw_rad": 0.05,
        },
    ),
    (
        "joint_margin_report",
        {
            "available": False,
            "ok": None,
            "reason": "joint_margin_unavailable",
        },
    ),
    ("dynamics_report", RuntimeError("dynamics reporter unavailable")),
)


@pytest.mark.parametrize(
    ("lock_trunk", "expected_kind"),
    [
        (True, "attached_arm"),
        (False, "arm"),
    ],
)
def test_guarded_ik_trunk_lock_selects_generator_without_collision_admission(
    lock_trunk,
    expected_kind,
):
    backend = RealCuroboBackend(None)
    captured = {}

    def compute(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    backend._compute_arm_plan = compute
    result = backend.plan_guarded_ik_step(
        hand="left",
        target_xyz=np.zeros(3),
        target_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        timeout_s=1.0,
        lock_trunk=lock_trunk,
    )

    assert result["ok"] is True
    assert captured["generator_kind"] == expected_kind
    assert "ik_world_collision_check" not in captured


def test_guarded_candidate_graph_does_not_query_collision_reporters():
    backend = RealCuroboBackend(None)
    assert not hasattr(backend, "_check_q_self_collisions")
    backend._curobo_eef_positions = lambda _generator, q: np.column_stack(
        [
            np.asarray(q)[:, 0],
            np.zeros(len(q)),
            np.zeros(len(q)),
        ]
    )

    selected, report = backend._select_guarded_candidate_path(
        object(),
        current_q=np.zeros((1, 2), dtype=np.float32),
        candidate_q_sets=[[[0.02, 0.0]]],
        candidate_selectable_masks=[[True]],
    )

    np.testing.assert_allclose(selected, [[0.02, 0.0]])
    assert report["selected"] is True
    assert report["layers"][0]["accepted_edges"] == 1


class _FakeEnv:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self._last_info = {"done": {"success": False}}
        self._gripper_latch = {"left": 1.0, "right": 1.0}

    def chunk_step(self, actions):
        self.calls.append(np.asarray(actions).copy())
        self.backend.advance()
        return (
            None,
            0.0,
            False,
            False,
            {
                **self._last_info,
                "_rpent": {"executed_steps": 1},
            },
        )


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
        self.joint_positions = np.zeros(28, dtype=np.float32)
        self.whole_body_plan_calls = []
        self.whole_body_hold_calls = []
        self.whole_body_tracking_calls = []
        self.isolation_capture_calls = []
        self.isolation_report_calls = []

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

    def plan_whole_body_trajectory(
        self,
        *,
        hand,
        target_xyz,
        target_quat_xyzw,
        timeout_s,
        attached_obj=None,
    ):
        self.target = np.asarray(target_xyz, dtype=np.float64)
        self.target_quat = (
            None
            if target_quat_xyzw is None
            else np.asarray(target_quat_xyzw, dtype=np.float64)
        )
        self.planned_targets.append(self.target.copy())
        self.attached_used.append(attached_obj)
        attachments_by_hand = {
            side: self.get_attached_object(side) for side in ("left", "right")
        }
        self.whole_body_plan_calls.append(
            {
                "hand": hand,
                "target_xyz": self.target.copy(),
                "target_quat_xyzw": (
                    None if self.target_quat is None else self.target_quat.copy()
                ),
                "timeout_s": float(timeout_s),
                "selected_attachment": attached_obj,
                "attachments_by_hand": attachments_by_hand,
            }
        )
        waypoint_count = 3
        q_dimension = 27 if self.bad_actions else 28
        trajectory = np.repeat(
            self.joint_positions[:q_dimension].reshape(1, -1),
            waypoint_count,
            axis=0,
        ).astype(np.float32)
        if self.bad_velocity:
            trajectory[1, 6:10] = 6.1
            trajectory[1, 10:17] = 6.1
            trajectory[1, 19:26] = 6.1
        certificate = {
            "schema_version": 1,
            "trajectory_sha256": hashlib.sha256(
                np.ascontiguousarray(trajectory, dtype=np.float32).tobytes()
            ).hexdigest(),
            "start_q_sha256": hashlib.sha256(
                np.ascontiguousarray(self.joint_positions, dtype=np.float32).tobytes()
            ).hexdigest(),
            "waypoint_count": waypoint_count,
            "q_dimension": q_dimension,
            "active_dof_count": 21,
            "selected_eef_goal_count": 1,
            "inactive_eef_goal_count": 0,
            "attachment_hand_count": 2,
            "world_collision_check": True,
            "self_collision_check": True,
            "post_interpolation_check": True,
            "collision_free_waypoints": waypoint_count,
        }
        collision_admission = {
            "available": True,
            "admitted": True,
            "world_collision_check": True,
            "self_collision_check": True,
            "obstacle_update": True,
            "full_trajectory": True,
            "post_interpolation_check": True,
            "colliding_waypoint_count": 0,
        }
        return {
            "ok": True,
            "joint_trajectory": trajectory,
            "whole_body_certificate": certificate,
            "expected_attachments_by_hand": attachments_by_hand,
            "metrics": {
                "motion_scope": "whole_body",
                "generator_kind": "whole_body",
                "active_dof_count": 21,
                "trajectory_waypoints": waypoint_count,
                "attachments_by_hand": {
                    side: {"available": value is not None}
                    for side, value in attachments_by_hand.items()
                },
                "collision_admission": collision_admission,
                "whole_body_certificate": certificate,
            },
        }

    def get_eef_pose(self, hand):
        return self.pose.copy(), self.quat.copy()

    def get_joint_positions(self):
        return self.joint_positions.copy()

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

    def capture_whole_body_contact_baseline(
        self, *, expected_attachments_by_hand
    ):
        del expected_attachments_by_hand
        if self.contact_mode == "unavailable":
            return {"available": False, "reason": "fake_contact_api_unavailable"}
        return {"available": True, "pairs": [], "pair_count": 0}

    def whole_body_contact_report(
        self, *, baseline, expected_attachments_by_hand
    ):
        del baseline, expected_attachments_by_hand
        if self.contact_mode == "unavailable":
            return {
                "available": False,
                "reason": "fake_contact_api_unavailable",
                "unexpected_contact": False,
            }
        return {
            "available": True,
            "unexpected_contact": self.contact_mode == "unexpected",
            "unexpected_pairs": (
                [["/World/robot/arm", "/World/floor"]]
                if self.contact_mode == "unexpected"
                else []
            ),
        }

    def get_attached_object(self, hand):
        if not isinstance(self.attached_obj, dict):
            return None
        eef_link = f"{hand}_eef_link"
        if eef_link not in self.attached_obj:
            return None
        return {eef_link: self.attached_obj[eef_link]}

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
        target = np.asarray(target_q, dtype=np.float32).reshape(-1)
        if hand is None:
            self.whole_body_tracking_calls.append(target.copy())
        return {
            "available": True,
            "reached": True,
            "max_articulation_error_rad": 0.0,
            "max_base_xy_error_m": 0.0,
            "base_yaw_error_rad": 0.0,
        }

    def capture_trajectory_hold_reference(
        self,
        *,
        hand,
        motion_scope="arm_only",
    ):
        if motion_scope == "whole_body":
            assert hand is None
            reference = {
                "hand": None,
                "motion_scope": "whole_body",
                "token": "whole_body_fixed_at_trajectory_start",
            }
            self.whole_body_hold_calls.append(reference)
            return reference
        return None

    def joint_target_to_action(
        self,
        target_q,
        *,
        hand,
        fixed_reference=None,
    ):
        target = np.asarray(target_q, dtype=np.float32).reshape(-1)
        if target.shape != (28,):
            raise ValueError(
                f"fake R1Pro whole-body q target must have shape (28,), got {target.shape}"
            )
        if fixed_reference is not None:
            assert hand is None
            assert fixed_reference == {
                "hand": None,
                "motion_scope": "whole_body",
                "token": "whole_body_fixed_at_trajectory_start",
            }
        action = self.hold.copy()
        action[ENV_ACTION_SEGMENTS["trunk"]] = target[6:10]
        action[ENV_ACTION_SEGMENTS["left_arm"]] = target[10:17]
        action[ENV_ACTION_SEGMENTS["right_arm"]] = target[19:26]
        return action

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

    def capture_single_arm_isolation_reference(self, *, hand, gripper_only):
        inactive = "right" if hand == "left" else "left"
        reference = {
            "selected_hand": hand,
            "gripper_only": bool(gripper_only),
            "mode": "gripper_only" if gripper_only else "arm_motion",
            "context_id": f"isolation-{len(self.isolation_capture_calls) + 1}",
            "reference_origin": "primitive_call_start",
            "inactive_hand": inactive,
            "inactive_attachment": self.get_attached_object(inactive),
        }
        if not gripper_only:
            reference["selected_attachment"] = self.get_attached_object(hand)
        self.isolation_capture_calls.append(reference)
        return reference

    def single_arm_isolation_report(
        self,
        *,
        hand,
        action,
        reference,
        gripper_only,
    ):
        assert hand == reference["selected_hand"]
        assert bool(gripper_only) is bool(reference["gripper_only"])
        assert np.asarray(action).shape == (23,)
        inactive = reference["inactive_hand"]
        inactive_actual = self.get_attached_object(inactive)
        inactive_expected = reference["inactive_attachment"]
        if inactive_actual is None or inactive_expected is None:
            inactive_matches = inactive_actual is None and inactive_expected is None
        else:
            inactive_matches, _identity = _attachment_identity_status(
                inactive_actual,
                inactive_expected,
                hand=inactive,
            )
        report = {
            "available": True,
            "ok": True,
            "selected_hand": hand,
            "mode": "gripper_only" if gripper_only else "arm_motion",
            "context_id": reference["context_id"],
            "reference_origin": reference["reference_origin"],
            "checks": {
                "locked_joints": {"available": True, "ok": True},
                "inactive_eef": {"available": True, "ok": True},
                "locked_gripper_commands": {"available": True, "ok": True},
                "inactive_attachment": {
                    "available": True,
                    "ok": inactive_matches,
                    "hand": inactive,
                    "expected_attached": inactive_expected is not None,
                    "actual_attached": inactive_actual is not None,
                    "matches": inactive_matches,
                },
            },
            "max_observed": {
                "base_position_m": 0.0,
                "base_orientation_rad": 0.0,
                "trunk_joint_rad": 0.0,
                "inactive_arm_joint_rad": 0.0,
                "inactive_eef_position_m": 0.0,
                "inactive_eef_orientation_rad": 0.0,
                "locked_gripper_command": 0.0,
            },
            "thresholds": {
                "base_position_m": 0.01,
                "base_orientation_rad": np.deg2rad(1.0),
                "joint_rad": 0.01,
                "inactive_eef_position_m": 0.01,
                "inactive_eef_orientation_rad": np.deg2rad(1.0),
                "locked_gripper_command": 1e-6,
            },
        }
        if not gripper_only:
            selected_actual = self.get_attached_object(hand)
            selected_expected = reference["selected_attachment"]
            if selected_actual is None or selected_expected is None:
                selected_matches = selected_actual is None and selected_expected is None
            else:
                selected_matches, _identity = _attachment_identity_status(
                    selected_actual,
                    selected_expected,
                    hand=hand,
                )
            report["checks"]["selected_attachment"] = {
                "available": True,
                "ok": selected_matches,
                "hand": hand,
                "expected_attached": selected_expected is not None,
                "actual_attached": selected_actual is not None,
                "matches": selected_matches,
            }
            report["ok"] = bool(report["ok"] and selected_matches)
        report["ok"] = bool(report["ok"] and inactive_matches)
        self.isolation_report_calls.append(report)
        return report


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


def _install_counted_reporter(backend, name, outcome):
    calls = []

    def reporter(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    setattr(backend, name, reporter)
    return calls


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


def test_base_planner_reports_unreachable_without_collision_classification():
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
                "reachable_count": 0,
            }
            return []

    result = Backend().plan_base_trajectory(
        hand="left",
        target_xyz=np.array([1.0, 0.0, 0.8]),
        standoff_m=0.85,
        timeout_s=1.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "navigation_unreachable"
    assert "collision" not in result["metrics"]["reason"]
    assert result["metrics"]["candidate_summary"]["traversable_count"] == 3


def test_move_to_rejects_bad_action_shape_before_env_step():
    executor, env = _executor(_FakeBackend(bad_actions=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "ValueError" in result["diagnostics"]["error"]
    assert env.calls == []


def test_move_to_stops_bounded_when_target_tolerance_is_not_met():
    executor, env = _executor(_FakeBackend(progress=False))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0], timeout_s=10)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "replan_no_progress"
    assert result["recoverable"] is True
    assert 20 <= len(env.calls) <= 30
    assert len(result["metrics"]["replan_rounds"]) == 2


@pytest.mark.parametrize(
    ("second_position_error", "second_tracking_error"),
    [
        (0.008, 0.99),
        (0.01, 0.95),
    ],
)
def test_move_to_replans_after_exact_position_or_tracking_progress_threshold(
    second_position_error,
    second_tracking_error,
):
    backend = _FakeBackend(progress=False)
    executor, _env = _executor(backend)
    scripted = iter(
        [
            (False, "stalled_tracking", 0.01, 1.0),
            (
                False,
                "stalled_tracking",
                second_position_error,
                second_tracking_error,
            ),
            (True, "reached", 0.0, 0.0),
        ]
    )

    def execute(*_args, **_kwargs):
        success, stop_reason, position_error, tracking_error = next(scripted)
        return {
            "primitive_success": success,
            "task_success": False,
            "stop_reason": stop_reason,
            "recoverable": True,
            "suggested_next_tool": None,
            "metrics": {
                "final_position_error_m": position_error,
                "final_joint_tracking": {
                    "normalized_21d_tracking_error": tracking_error,
                },
                "trajectory_complete": success,
            },
            "diagnostics": {"trace": []},
        }

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert len(backend.whole_body_plan_calls) == 3
    assert len(result["metrics"]["replan_rounds"]) == 3


@pytest.mark.parametrize(
    ("second_position_error", "second_tracking_error"),
    [
        (0.008001, 0.99),
        (0.01, 0.950001),
        (float("nan"), 0.99),
        (0.01, float("nan")),
    ],
)
def test_move_to_replan_progress_must_be_finite_and_reach_the_exact_threshold(
    second_position_error,
    second_tracking_error,
):
    backend = _FakeBackend(progress=False)
    executor, env = _executor(backend)
    scripted = iter(
        [
            (0.01, 1.0),
            (second_position_error, second_tracking_error),
        ]
    )

    def execute(*_args, **_kwargs):
        position_error, tracking_error = next(scripted)
        return {
            "primitive_success": False,
            "task_success": False,
            "stop_reason": "stalled_tracking",
            "recoverable": True,
            "suggested_next_tool": None,
            "metrics": {
                "final_position_error_m": position_error,
                "final_joint_tracking": {
                    "normalized_21d_tracking_error": tracking_error,
                },
                "trajectory_complete": False,
            },
            "diagnostics": {"trace": []},
        }

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "replan_no_progress"
    assert len(backend.whole_body_plan_calls) == 2
    assert env.calls == []


@pytest.mark.parametrize(
    ("contact_mode", "expected_stop", "expected_env_actions"),
    [
        ("unexpected", "unexpected_contact", 1),
        ("unavailable", "contact_feedback_unavailable", 0),
    ],
)
def test_move_to_contact_failures_stop_without_replanning(
    contact_mode,
    expected_stop,
    expected_env_actions,
):
    backend = _FakeBackend(contact_mode=contact_mode)
    executor, env = _executor(backend)

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == expected_stop
    assert len(backend.whole_body_plan_calls) == 1
    assert len(env.calls) == expected_env_actions


def test_held_move_uses_call_start_attachment_in_whole_body_plan():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__(attached_obj={"left_eef_link": self_root})
            self.attachment_reads = 0

        def get_attached_object(self, hand):
            if hand == "right":
                return None
            assert hand == "left"
            self.attachment_reads += 1
            return self.attached_obj

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
    assert backend.attachment_reads >= 3
    assert len(backend.whole_body_plan_calls) == 1
    assert backend.whole_body_plan_calls[0]["selected_attachment"] is captured
    assert backend.whole_body_plan_calls[0]["attachments_by_hand"]["left"] is captured
    assert backend.attached_used[-1] is captured
    assert result["metrics"]["attachments_by_hand"] == {
        "left": {"available": True},
        "right": {"available": False},
    }


def test_held_move_validates_exact_attachment_identity_after_every_step():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"left_eef_link": self.target_root}
            self.attachment_reads = 0

        def get_attached_object(self, hand):
            if hand == "right":
                return None
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

    result = executor.move_to(hand="left", target_xyz=[0.45, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert backend.attachment_reads >= len(env.calls) + 2
    assert (
        result["metrics"]["whole_body_execution"][
            "dual_attachment_checked_each_nonterminal_step"
        ]
        is True
    )
    assert (
        result["metrics"]["whole_body_execution"][
            "raw_success_preempts_post_step_safety_checks"
        ]
        is True
    )
    assert "single_arm_isolation" not in result["metrics"]


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
    assert result["stop_reason"] == "attachment_identity_mismatch"
    assert len(env.calls) == 2
    assert result["metrics"]["whole_body_attachment"]["hand"] == "left"
    assert result["metrics"]["whole_body_attachment"]["matches"] is False


def test_held_move_fails_bounded_on_attachment_identity_mismatch():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"right_eef_link": self.target_root}

        def get_attached_object(self, hand):
            return self.attached_obj if hand == "left" else None

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
    result = BehaviorToolResult(name="observe", result=payload)

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
    reporter_calls = {
        name: _install_counted_reporter(
            backend,
            name,
            RuntimeError(f"{name} must not participate in press execution"),
        )
        for name in ("collision_report", "joint_margin_report", "dynamics_report")
    }
    executor, _ = _executor(backend)
    executor._gripper_command = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("press must not close the gripper")
    )

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.004,
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
    assert all(calls == [] for calls in reporter_calls.values())


def test_default_press_precontact_stays_outside_curobo_activation_zone():
    backend = _FakeBackend(contact_mode="expected_near_target")
    executor, _ = _executor(backend)
    surface = np.array([0.5, 0.0, -0.1])

    result = executor.press(
        hand="left",
        target_xyz=surface,
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.03,
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
    assert report["terminal_max_command_acceleration_proxy_rad_s2"] < 15.0


def test_press_uses_collision_certified_whole_body_receding_horizon_steps():
    class RecedingHorizonBackend(_FakeBackend):
        def __init__(self):
            super().__init__(contact_mode="expected_near_target")
            self.guarded_step_calls = 0

        def plan_whole_body_trajectory(self, **kwargs):
            self.guarded_step_calls += 1
            return super().plan_whole_body_trajectory(**kwargs)

    backend = RecedingHorizonBackend()
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.004,
    )

    assert result["primitive_success"] is True
    assert backend.guarded_step_calls > 1
    assert (
        result["metrics"]["guarded_execution_mode"] == "receding_horizon_cartesian_ik"
    )
    assert result["metrics"]["motion_scope"] == "whole_body"
    assert (
        result["metrics"]["whole_body_execution"][
            "collision_certificate_verified_before_each_guarded_action"
        ]
        is True
    )


def test_guarded_motion_does_not_treat_contact_as_a_runtime_safety_gate():
    backend = _FakeBackend(contact_mode="expected")
    executor, env = _executor(backend)

    result = executor._guarded_incremental_move(
        hand="left",
        target_xyz=np.array([0.5, 0.0, -0.04]),
        target_quat_xyzw=None,
        direction=np.array([0.0, 0.0, -1.0]),
        position_tolerance_m=0.015,
        timeout_s=45.0,
    )

    assert result["primitive_success"] is True
    assert env.calls


def test_guarded_contact_api_unavailable_is_structured_failure():
    backend = _FakeBackend(contact_mode="unavailable")
    executor, _ = _executor(backend)

    result = executor.press(
        hand="left",
        target_xyz=[0.5, 0.0, -0.01],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.004,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "contact_feedback_unavailable"
    assert result["metrics"]["whole_body_contact_baseline"]["available"] is False


def test_move_to_returns_after_first_stable_endpoint_without_extra_hold():
    executor, env = _executor(_FakeBackend(progress=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert result["metrics"]["held_steps"] == 1
    assert len(env.calls) >= 1


def test_move_to_does_not_reject_command_delta_dynamics_heuristic():
    executor, env = _executor(_FakeBackend(bad_velocity=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert env.calls


@pytest.mark.parametrize(
    ("reporter_name", "outcome"),
    _REMOVED_RUNTIME_REPORTER_CASES,
    ids=(
        "collision_true",
        "collision_unavailable",
        "joint_margin_false",
        "joint_margin_unavailable",
        "dynamics_exception",
    ),
)
def test_move_to_execution_does_not_query_removed_runtime_safety_reporters(
    reporter_name,
    outcome,
):
    backend = _FakeBackend()
    calls = _install_counted_reporter(backend, reporter_name, outcome)
    executor, env = _executor(backend)

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert env.calls
    assert calls == []


def test_rotate_wrist_does_not_query_removed_runtime_safety_reporters():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()

        @staticmethod
        def collision_report(*_args, **_kwargs):
            raise AssertionError("rotate_wrist must not query collision telemetry")

        @staticmethod
        def joint_margin_report(*_args, **_kwargs):
            raise AssertionError("rotate_wrist must not query joint-margin telemetry")

    backend = Backend()
    backend.dynamics_report = lambda: (_ for _ in ()).throw(
        AssertionError("rotate_wrist must not query dynamics telemetry")
    )
    executor, env = _executor(backend)

    result = executor.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 0.0, 1.0, np.pi / 8],
    )

    assert result["primitive_success"] is True
    assert env.calls


def test_public_planner_timeout_remains_a_runtime_gate():
    executor, env = _executor(_FakeBackend(progress=False))
    chunk_step = env.chunk_step

    def slow_chunk_step(actions):
        time.sleep(0.02)
        return chunk_step(actions)

    env.chunk_step = slow_chunk_step
    started = time.monotonic()
    result = executor.move_to(
        hand="left",
        target_xyz=[0.0, 0.0, 0.0],
        timeout_s=0.01,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "execution_budget_exhausted"
    assert time.monotonic() - started < 0.1


def test_move_to_clamps_public_planning_and_execution_wall_clock_budgets(
    monkeypatch,
):
    deadlines = []

    @contextmanager
    def recording_deadline(timeout_s, operation):
        deadlines.append((str(operation), float(timeout_s)))
        yield

    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        recording_deadline,
    )
    backend = _FakeBackend(progress=True)
    executor, _env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.0, 0.0, 0.0],
        timeout_s=999.0,
    )

    assert result["primitive_success"] is True
    assert deadlines[0] == ("planner tool move_to", pytest.approx(240.0))
    planning_deadlines = [
        timeout
        for operation, timeout in deadlines
        if operation == "whole-body planning transaction"
    ]
    execution_deadlines = [
        timeout
        for operation, timeout in deadlines
        if operation == "whole-body execution transaction"
    ]
    assert planning_deadlines
    assert execution_deadlines
    assert max(planning_deadlines) <= 60.0
    assert max(execution_deadlines) <= 180.0
    assert backend.whole_body_plan_calls[0]["timeout_s"] <= 60.0
    assert result["metrics"]["total_deadline_s"] == pytest.approx(240.0)
    assert result["metrics"]["planning_hard_limit_s"] == pytest.approx(60.0)
    assert result["metrics"]["execution_hard_limit_s"] == pytest.approx(180.0)


def test_move_to_public_defaults_are_20mm_5deg_and_240s():
    signature = inspect.signature(PlannerExecutor.move_to)

    assert signature.parameters["position_tolerance_m"].default == pytest.approx(
        0.02
    )
    assert signature.parameters["orientation_tolerance_rad"].default == pytest.approx(
        np.deg2rad(5.0)
    )
    assert signature.parameters["timeout_s"].default == pytest.approx(240.0)


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


@pytest.mark.parametrize("representation", ["active_only", "already_full"])
def test_base_ik_merge_never_augments_or_writes_locked_joints(
    tmp_path,
    representation,
):
    full_names = [
        "base_footprint_x_joint",
        "base_footprint_y_joint",
        "base_footprint_z_joint",
        "base_footprint_rx_joint",
        "base_footprint_ry_joint",
        "base_footprint_rz_joint",
        *[f"torso_joint{i}" for i in range(1, 5)],
        *[f"left_arm_joint{i}" for i in range(1, 8)],
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        *[f"right_arm_joint{i}" for i in range(1, 8)],
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    ]
    lock_names = [
        name for name in full_names if name not in BASE_ACTIVE_JOINT_NAMES
    ]
    lock_state = type("LockState", (), {"joint_names": lock_names})()
    config = type("Config", (), {"lock_jointstate": lock_state})()
    kinematics = type(
        "Kinematics",
        (),
        {
            "joint_names": list(BASE_ACTIVE_JOINT_NAMES),
            "kinematics_config": config,
        },
    )()
    motion_generator = type("MotionGenerator", (), {"kinematics": kinematics})()
    embodiment = type("Embodiment", (), {"BASE": "base"})
    generator = type(
        "Generator",
        (),
        {
            "robot_joint_names": full_names,
            "mg": {"base": motion_generator},
        },
    )()
    path_names = (
        list(BASE_ACTIVE_JOINT_NAMES)
        if representation == "active_only"
        else full_names
    )
    path = type(
        "Path",
        (),
        {
            "joint_names": path_names,
            "position": np.arange(
                100.0,
                100.0 + len(path_names),
                dtype=np.float32,
            ),
        },
    )()
    robot = type("Robot", (), {"joints": dict.fromkeys(full_names)})()
    call_start = np.arange(28, dtype=np.float32)
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._embodiment_cls = embodiment

    merged, report = backend._merge_base_ik_solution_into_full_q(
        generator,
        robot,
        path,
        call_start_q=call_start,
    )

    full_index = {name: index for index, name in enumerate(full_names)}
    path_index = {name: index for index, name in enumerate(path_names)}
    for name in BASE_ACTIVE_JOINT_NAMES:
        assert merged[0, full_index[name]] == pytest.approx(
            path.position[path_index[name]]
        )
    for name in lock_names:
        assert merged[0, full_index[name]] == pytest.approx(
            call_start[full_index[name]]
        )
    assert report["source_representation"] == representation
    assert report["get_full_js_called"] is False
    assert report["locked_joint_count"] == 25


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


def test_quarantined_generator_is_rebuilt_without_collision_warmup(tmp_path):
    class Generator:
        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    old = object()
    backend._generators["arm:left"] = old
    quarantined = backend._quarantine_generator(
        kind="arm", hand="left", reason="TimeoutError: injected"
    )
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = type("Embodiment", (), {"DEFAULT": "default"})
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )

    rebuilt = backend._generator(kind="arm", hand="left")

    assert quarantined["requires_fresh_rebuild"] is True
    assert rebuilt is not old
    assert rebuilt.kwargs["motion_cfg_kwargs"]["self_collision_check"] is False
    assert "arm:left" not in backend._invalid_generators


def test_generator_rebuild_does_not_call_collision_warmup(tmp_path):
    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._invalid_generators.add("arm:left")
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = type("Embodiment", (), {"DEFAULT": "default"})
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )

    rebuilt = backend._generator(kind="arm", hand="left")

    assert rebuilt is backend._generators["arm:left"]
    assert "arm:left" not in backend._invalid_generators


def test_generator_construction_does_not_probe_null_lock_resolution(tmp_path):
    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def update_locked_joints(*_args, **_kwargs):
            raise AssertionError(
                "generator construction must not probe lock resolution"
            )

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = type("Embodiment", (), {"DEFAULT": "default"})
    backend._find_robot = lambda: object()
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )

    assert isinstance(backend._generator(kind="arm", hand="right"), Generator)


@_REQUIRES_TORCH
def test_fresh_attached_generator_refreshes_world_registry_without_collision_admission(
    tmp_path,
):
    class Generator:
        batch_size = 2

        def __init__(self):
            self.world_ready = False
            self.world_updates = 0
            self.compute_kwargs = None

        def update_obstacles(self):
            self.world_updates += 1
            self.world_ready = True

        def compute_trajectories(self, _positions, _quaternions, **kwargs):
            assert self.world_ready, "fresh attached mesh registry was not initialized"
            self.compute_kwargs = kwargs
            return np.array([True, False]), [object(), None]

        @staticmethod
        def check_collisions(*_args, **_kwargs):
            raise AssertionError("collision booleans must not be queried")

    class Robot:
        @staticmethod
        def get_joint_positions():
            return np.zeros(4, dtype=np.float32)

    generator = Generator()
    attached = {"left_eef_link": object()}
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: Robot()
    backend._hand_config_path = lambda _hand, lock_trunk=False: (
        tmp_path / ("attached.yaml" if lock_trunk else "arm.yaml")
    )
    backend._merge_ik_solution_into_full_q = lambda *_args, **_kwargs: (
        np.array([[0.2, 0.1, 0.0, 0.0]], dtype=np.float32),
        {"source": "test"},
    )

    result = backend.plan_attached_arm_trajectory(
        hand="left",
        target_xyz=np.array([0.1, 0.0, 0.8]),
        target_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        timeout_s=1.0,
        attached_obj=attached,
    )

    assert result["ok"] is True
    assert generator.world_updates == 1
    assert generator.compute_kwargs["attached_obj"] is attached
    assert generator.compute_kwargs["ik_only"] is True
    assert generator.compute_kwargs["ik_world_collision_check"] is False
    assert generator.compute_kwargs["skip_obstacle_update"] is True
    assert result["metrics"]["world_mesh_registry_refreshed"] is True
    assert result["metrics"]["collision_admission_enabled"] is False


def test_real_warmup_uses_only_whole_body_planners_without_env_actions(tmp_path):
    class Robot:
        @staticmethod
        def get_joint_positions():
            return np.zeros(4, dtype=np.float32)

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._find_robot = lambda: Robot()
    backend._base_xy_yaw = lambda _robot: np.zeros(4)
    eef_poses = {
        "left": (
            np.asarray([0.41, 0.22, 0.83]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        ),
        "right": (
            np.asarray([0.39, -0.24, 0.81]),
            np.asarray([0.0, 0.0, 0.1, np.sqrt(0.99)]),
        ),
    }
    backend.get_eef_pose = lambda hand: eef_poses[hand]
    assert not hasattr(backend, "_candidate_base_collision_reports")
    assert not hasattr(backend, "_check_q_trajectory_collisions")
    backend._compute_base_plan = lambda **_kwargs: pytest.fail(
        "whole-body analytic warmup must not compile the unrelated BASE planner"
    )
    arm_calls = []

    def arm_plan(**kwargs):
        arm_calls.append(kwargs)
        return {"ok": True}

    whole_body_calls = []
    backend.plan_whole_body_trajectory = lambda **kwargs: (
        whole_body_calls.append(kwargs) or {"ok": True}
    )

    result = backend.warmup()

    assert result["status"] == "complete"
    assert arm_calls == []
    assert [call["hand"] for call in whole_body_calls] == ["left", "right"]
    for call in whole_body_calls:
        expected_position, expected_quaternion = eef_poses[call["hand"]]
        np.testing.assert_allclose(call["target_xyz"], expected_position)
        np.testing.assert_allclose(
            call["target_quat_xyzw"],
            expected_quaternion,
        )
        assert call["timeout_s"] == pytest.approx(60.0)
    assert result["identity_warmup"]["env_actions_sent"] == 0
    assert result["identity_warmup"]["simulator_advanced"] is False
    assert [sample["query"] for sample in result["identity_warmup"]["hands"]] == [
        "identity_trajectory",
        "identity_trajectory",
    ]


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


@pytest.mark.parametrize("hand", ["left", "right"])
def test_gripper_close_preserves_current_hold_segments(hand):
    backend = _FakeBackend()
    executor, env = _executor(backend)
    inactive = "right" if hand == "left" else "left"

    result = executor._gripper_command(hand, opening=0.0, timeout_s=30.0)

    assert result["primitive_success"] is True
    commands = np.asarray(
        [call[0, ENV_ACTION_SEGMENTS[f"{hand}_gripper"]][0] for call in env.calls]
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
    for call in env.calls:
        action = call[0]
        for segment_name in (
            "base",
            "trunk",
            "left_arm",
            "right_arm",
            f"{inactive}_gripper",
        ):
            segment = ENV_ACTION_SEGMENTS[segment_name]
            np.testing.assert_allclose(action[segment], backend.hold[segment])


@pytest.mark.parametrize("hand", ["left", "right"])
def test_held_close_requires_ten_successful_endpoint_attachment_steps(hand):
    backend = _FakeBackend()
    eef_link = f"{hand}_eef_link"
    backend.attached_obj = {eef_link: backend.target_root}
    executor, env = _executor(backend)
    env._gripper_latch[hand] = -1.0
    expected = {eef_link: backend.target_root}

    result = executor._gripper_command(
        hand,
        opening=0.0,
        timeout_s=30.0,
        hold_steps_required=10,
        expected_attachment=expected,
        require_attachment=True,
    )

    assert result["primitive_success"] is True
    assert result["metrics"]["attachment_endpoint_held_steps"] == 10
    assert result["metrics"]["attachment_confirmation_steps"] >= 10
    assert len(env.calls) == 12
    assert env._gripper_latch[hand] == pytest.approx(-1.0)
    endpoint_commands = [
        call[0, ENV_ACTION_SEGMENTS[f"{hand}_gripper"]][0] for call in env.calls[-10:]
    ]
    np.testing.assert_allclose(endpoint_commands, -1.0)


@pytest.mark.parametrize(
    ("opening", "initial_latch", "expected_steps"),
    [
        (0.0, 1.0, 180),
        (1.0, -1.0, 3),
    ],
)
@pytest.mark.parametrize(
    ("reporter_name", "outcome"),
    _REMOVED_RUNTIME_REPORTER_CASES,
    ids=(
        "collision_true",
        "collision_unavailable",
        "joint_margin_false",
        "joint_margin_unavailable",
        "dynamics_exception",
    ),
)
def test_gripper_commands_ignore_removed_runtime_safety_reporters(
    opening,
    initial_latch,
    expected_steps,
    reporter_name,
    outcome,
):
    backend = _FakeBackend()
    calls = _install_counted_reporter(backend, reporter_name, outcome)
    executor, env = _executor(backend)
    env._gripper_latch["left"] = initial_latch

    result = executor._gripper_command("left", opening=opening, timeout_s=30.0)

    assert result["primitive_success"] is True
    assert len(env.calls) == expected_steps
    assert env._gripper_latch["left"] == pytest.approx(-1.0 if opening < 0.5 else 1.0)
    assert calls == []


def test_gripper_collision_telemetry_does_not_interrupt_receipted_latch_progress():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.collision_calls = 0

        def collision_report(self):
            self.collision_calls += 1
            return {
                "available": True,
                "colliding": True,
                "min_margin_m": -0.001,
                "margin_available": True,
            }

    backend = Backend()
    executor, env = _executor(backend)

    result = executor._gripper_command("left", opening=0.0, timeout_s=30.0)

    assert result["primitive_success"] is True
    assert len(env.calls) == 180
    assert env._gripper_latch["left"] == pytest.approx(-1.0)
    assert backend.collision_calls == 0


def test_gripper_step_exception_does_not_mutate_latch():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    def fail_step(_actions):
        raise RuntimeError("step failed before command acceptance")

    env.chunk_step = fail_step

    with pytest.raises(RuntimeError, match="step failed before command acceptance"):
        executor._gripper_command("left", opening=0.0, timeout_s=30.0)

    assert env._gripper_latch["left"] == pytest.approx(1.0)


def test_gripper_zero_step_response_does_not_mutate_latch():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    def zero_step(_actions):
        return (
            None,
            None,
            False,
            True,
            {"done": {"success": False}, "_rpent": {"executed_steps": 0}},
        )

    env.chunk_step = zero_step

    with pytest.raises(RuntimeError, match="not executed exactly once"):
        executor._gripper_command("left", opening=0.0, timeout_s=30.0)

    assert env._gripper_latch["left"] == pytest.approx(1.0)


def test_gripper_missing_execution_receipt_does_not_mutate_latch():
    backend = _FakeBackend()
    executor, env = _executor(backend)
    env.chunk_step = lambda _actions: None

    with pytest.raises(RuntimeError, match="did not return an execution receipt"):
        executor._gripper_command("left", opening=0.0, timeout_s=30.0)

    assert env._gripper_latch["left"] == pytest.approx(1.0)


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


def test_base_candidate_search_does_not_query_collision_reporters():
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
    assert not hasattr(backend, "_candidate_base_collision_reports")
    backend.check_candidate_arm_reachability = lambda **_kwargs: (
        True,
        "reachable_candidate",
        {
            "reachability_stage": "candidate_multi_orientation_ik",
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
    assert backend._last_base_candidate_summary["candidate_batch_size"] == 9
    assert backend._last_base_candidate_summary["candidate_limit"] == 6
    assert all("base_collision_report" not in candidate for candidate in ranked)


def test_base_reachability_uses_precontact_point_toward_candidate():
    target = np.array([2.0, 1.0, 0.8])
    candidate = np.array([1.0, 1.0, 0.0])

    reachability_target = RealCuroboBackend._candidate_reachability_target(
        target,
        candidate,
    )

    np.testing.assert_allclose(reachability_target, [1.85, 1.0, 0.8])


@_REQUIRES_TORCH
def test_base_plan_requires_dense_collision_admission(tmp_path):
    full_names = [
        "base_footprint_x_joint",
        "base_footprint_y_joint",
        "base_footprint_z_joint",
        "base_footprint_rx_joint",
        "base_footprint_ry_joint",
        "base_footprint_rz_joint",
        *[f"torso_joint{i}" for i in range(1, 5)],
        *[f"left_arm_joint{i}" for i in range(1, 8)],
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        *[f"right_arm_joint{i}" for i in range(1, 8)],
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    ]
    lock_names = [
        name for name in full_names if name not in BASE_ACTIVE_JOINT_NAMES
    ]
    lock_state = type("LockState", (), {"joint_names": lock_names})()
    config = type("Config", (), {"lock_jointstate": lock_state})()
    kinematics = type(
        "Kinematics",
        (),
        {
            "joint_names": list(BASE_ACTIVE_JOINT_NAMES),
            "kinematics_config": config,
        },
    )()
    motion_generator = type("MotionGenerator", (), {"kinematics": kinematics})()
    goal = np.zeros(28, dtype=np.float32)
    goal[0] = 0.2
    goal[1] = 0.1
    goal[5] = 0.2
    path = type(
        "Path",
        (),
        {"joint_names": full_names, "position": goal},
    )()

    class Generator:
        batch_size = 2
        robot_joint_names = full_names
        mg = {"base": motion_generator}

        def __init__(self):
            self.compute_kwargs = None

        def compute_trajectories(self, _positions, _quaternions, **kwargs):
            self.compute_kwargs = kwargs
            return np.array([True, False]), [path, None]

        @staticmethod
        def path_to_joint_trajectory(_path, **_kwargs):
            raise AssertionError("BASE full-js augmentation must not be called")

        @staticmethod
        def check_collisions(q, **kwargs):
            assert kwargs["self_collision_check"] is True
            assert kwargs["skip_obstacle_update"] is True
            return np.zeros(len(q), dtype=bool)

    class Robot:
        base_idx = np.arange(6)
        joints = dict.fromkeys(full_names)

        @staticmethod
        def get_joint_positions():
            return np.zeros(28, dtype=np.float32)

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: Robot()
    backend._embodiment_cls = type("Embodiment", (), {"BASE": "base"})
    backend._base_config_path = lambda: tmp_path / "base.yaml"
    backend._all_attached_objects = lambda **_kwargs: (
        None,
        {"left": None, "right": None},
    )

    result = backend._compute_base_plan(
        target_xyyaw=np.array([0.2, 0.1, 0.2]),
        timeout_s=1.0,
    )

    assert result["ok"] is True
    assert generator.compute_kwargs["ik_only"] is True
    assert generator.compute_kwargs["ik_world_collision_check"] is True
    assert generator.compute_kwargs["skip_obstacle_update"] is False
    assert result["metrics"]["collision_admission_enabled"] is True
    assert result["metrics"]["obstacle_update"] is True
    assert result["metrics"]["collision_admission"]["admitted"] is True
    assert result["metrics"]["base_trajectory_certificate"][
        "post_interpolation_check"
    ] is True
    assert len(result["joint_trajectory"]) >= 1


@_REQUIRES_TORCH
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
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )
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


@_REQUIRES_TORCH
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
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )
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


@_REQUIRES_TORCH
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
    backend._hand_config_path = lambda _hand, *, lock_trunk=False: (
        tmp_path / "official_arm.yaml"
    )
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


@_REQUIRES_TORCH
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


@_REQUIRES_TORCH
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


@_REQUIRES_TORCH
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

    assert report["reached"] is True
    assert report["base_waypoint_xy_tolerance_m"] == 0.02
    target[0] = 0.0201
    assert backend.joint_tracking_report(target, hand=None)["reached"] is False


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
    assert report["articulation_waypoint_tolerance_rad"] == 0.02
    target[10] = 0.0201
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
            current[17] = 0.0201
            return current

    backend = RealCuroboBackend(None)
    backend._robot = Robot()
    target = np.zeros(28, dtype=np.float32)

    left_report = backend.joint_tracking_report(target, hand="left")
    right_report = backend.joint_tracking_report(target, hand="right")

    assert left_report["reached"] is True
    assert left_report["max_articulation_error_rad"] == 0.0
    assert right_report["reached"] is False
    assert right_report["max_articulation_error_rad"] == pytest.approx(0.0201)


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

    planner = PlannerExecutor(
        env=object(),
        frame_cache=FrameCache(),
        output_dir=tmp_path,
        backend=Backend(),
    )
    assert planner.warmup() == {"status": "complete", "elapsed_s": 1.25}

    planner.backend = object()
    with pytest.raises(RuntimeError, match="safety warmup"):
        planner.warmup()


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


@pytest.mark.parametrize("hand", ["left", "right"])
def test_arm_q_trajectory_uses_one_fixed_hold_for_inactive_segments(hand):
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.fixed_hold = np.arange(23, dtype=np.float32) * 0.01

        def hold_action(self, hand=None):
            assert hand == selected_hand
            return self.fixed_hold.copy()

        def joint_target_to_action(self, target_q, *, hand):
            del target_q
            assert hand == selected_hand
            action = np.full(23, 9.0, dtype=np.float32)
            action[ENV_ACTION_SEGMENTS["trunk"]] = 2.0
            action[ENV_ACTION_SEGMENTS[f"{selected_hand}_arm"]] = 3.0
            return action

    selected_hand = hand
    inactive_hand = "right" if hand == "left" else "left"
    backend = Backend()
    executor, env = _executor(backend)
    result = executor._execute_actions(
        None,
        hand=hand,
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        joint_trajectory=np.zeros((3, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    assert len(env.calls) == 3
    for call in env.calls:
        action = call[0]
        for segment_name in (
            "base",
            "trunk",
            f"{inactive_hand}_arm",
            "left_gripper",
            "right_gripper",
        ):
            segment = ENV_ACTION_SEGMENTS[segment_name]
            np.testing.assert_allclose(action[segment], backend.fixed_hold[segment])
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS[f"{hand}_arm"]],
            3.0,
        )


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("gripper_only", [False, True])
def test_single_arm_isolation_mask_is_left_right_symmetric(hand, gripper_only):
    action = np.arange(23, dtype=np.float32) + 100.0
    hold = np.arange(23, dtype=np.float32) * 0.01
    inactive = "right" if hand == "left" else "left"

    isolated = _apply_single_arm_isolation_mask(
        action,
        hold,
        hand=hand,
        gripper_only=gripper_only,
    )

    active_segment = f"{hand}_gripper" if gripper_only else f"{hand}_arm"
    locked_segments = (
        (
            "base",
            "trunk",
            "left_arm",
            "right_arm",
            f"{inactive}_gripper",
        )
        if gripper_only
        else (
            "base",
            "trunk",
            f"{inactive}_arm",
            "left_gripper",
            "right_gripper",
        )
    )
    for segment_name in locked_segments:
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_array_equal(isolated[segment], hold[segment])
    np.testing.assert_array_equal(
        isolated[ENV_ACTION_SEGMENTS[active_segment]],
        action[ENV_ACTION_SEGMENTS[active_segment]],
    )


@pytest.mark.parametrize("hand", ["left", "right"])
def test_move_to_plans_directly_with_collision_certified_whole_body(hand):
    class Backend(_FakeBackend):
        @staticmethod
        def check_arm_reachability(**_kwargs):
            raise AssertionError("move_to must not run a redundant arm-only IK probe")

        @staticmethod
        def plan_arm_trajectory(**_kwargs):
            raise AssertionError("move_to must not enter the legacy arm-only planner")

    backend = Backend()
    executor, env = _executor(backend)

    result = executor.move_to(
        hand=hand,
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is True
    assert result["metrics"]["motion_scope"] == "whole_body"
    assert result["metrics"]["generator_kind"] == "whole_body"
    assert result["metrics"]["active_dof_count"] == 21
    assert result["metrics"]["collision_admission"] == {
        "available": True,
        "admitted": True,
        "world_collision_check": True,
        "self_collision_check": True,
        "obstacle_update": True,
        "full_trajectory": True,
        "post_interpolation_check": True,
        "colliding_waypoint_count": 0,
    }
    assert len(backend.whole_body_plan_calls) == 1
    assert backend.whole_body_hold_calls == [
        {
            "hand": None,
            "motion_scope": "whole_body",
            "token": "whole_body_fixed_at_trajectory_start",
        }
    ]
    assert backend.whole_body_tracking_calls
    assert env.calls
    for call in env.calls:
        action = call[0]
        np.testing.assert_array_equal(
            action[ENV_ACTION_SEGMENTS["base"]],
            backend.hold[ENV_ACTION_SEGMENTS["base"]],
        )
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]],
                backend.hold[ENV_ACTION_SEGMENTS[f"{side}_gripper"]],
            )


@pytest.mark.parametrize(
    "certificate_fault",
    ["collision_admission", "missing_certificate", "trajectory_digest"],
)
def test_move_to_requires_whole_body_collision_certificate_before_first_action(
    certificate_fault,
):
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            if certificate_fault == "collision_admission":
                result["metrics"]["collision_admission"]["admitted"] = False
            elif certificate_fault == "missing_certificate":
                result["whole_body_certificate"] = None
            else:
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "trajectory_sha256": "tampered",
                }
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(
        hand="left",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == (
        "error"
        if certificate_fault == "trajectory_digest"
        else "collision_admission_unavailable"
    )
    assert env.calls == []


@pytest.mark.parametrize("selected_hand", ["left", "right"])
def test_move_to_accepts_and_monitors_simultaneous_dual_attachments(selected_hand):
    left_root = object()
    right_root = object()
    backend = _FakeBackend(
        attached_obj={
            "left_eef_link": left_root,
            "right_eef_link": right_root,
        }
    )
    executor, env = _executor(backend)
    env._gripper_latch.update({"left": -1.0, "right": -1.0})

    result = executor.move_to(
        hand=selected_hand,
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is True
    assert result["metrics"]["attachments_by_hand"] == {
        "left": {"available": True},
        "right": {"available": True},
    }
    assert len(backend.whole_body_plan_calls) == 1
    snapshot = backend.whole_body_plan_calls[0]["attachments_by_hand"]
    assert snapshot["left"]["left_eef_link"] is left_root
    assert snapshot["right"]["right_eef_link"] is right_root
    assert backend.whole_body_plan_calls[0]["selected_attachment"][
        f"{selected_hand}_eef_link"
    ] is (left_root if selected_hand == "left" else right_root)
    assert env.calls


@pytest.mark.parametrize("changed_hand", ["left", "right"])
def test_move_to_rejects_either_attachment_identity_change_before_execution(
    changed_hand,
):
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            self.attached_obj[f"{changed_hand}_eef_link"] = object()
            return result

    backend = Backend(
        attached_obj={
            "left_eef_link": object(),
            "right_eef_link": object(),
        }
    )
    executor, env = _executor(backend)
    env._gripper_latch.update({"left": -1.0, "right": -1.0})

    result = executor.move_to(
        hand="left",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_identity_mismatch"
    assert result["metrics"]["attachment_identity"]["hand"] == changed_hand
    assert env.calls == []


def test_rotate_and_press_use_whole_body_planning_without_legacy_arm_fallback():
    class Backend(_FakeBackend):
        @staticmethod
        def check_arm_reachability(**_kwargs):
            raise AssertionError("whole-body Cartesian tools must skip arm-only IK")

        @staticmethod
        def plan_arm_with_trunk_trajectory(**_kwargs):
            raise AssertionError("legacy trunk-assist planner must not be called")

    rotate_backend = Backend()
    rotate_executor, rotate_env = _executor(rotate_backend)
    rotate = rotate_executor.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.1],
    )

    assert rotate["primitive_success"] is True
    assert rotate["metrics"]["motion_scope"] == "whole_body"
    assert len(rotate_backend.whole_body_plan_calls) == 1
    assert rotate_env.calls

    press_backend = Backend(contact_mode="expected_near_target")
    press_executor, press_env = _executor(press_backend)
    pressed = press_executor.press(
        hand="right",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.004,
    )

    assert pressed["primitive_success"] is True
    assert pressed["metrics"]["motion_scope"] == "whole_body"
    assert pressed["metrics"]["whole_body_execution"]["ok"] is True
    assert press_backend.whole_body_plan_calls
    assert press_env.calls


def test_navigation_planner_uses_full_official_path_and_locks_nonbase_q(tmp_path):
    shortest_path_calls: list[dict[str, object]] = []

    class TraversabilityMap:
        floor_map = [np.ones((16, 16), dtype=np.uint8)]
        floor_heights = [0.0]

        @staticmethod
        def world_to_map(_xy):
            return np.array([8, 8], dtype=np.int64)

        def get_shortest_path(
            self,
            floor,
            source,
            target,
            *,
            entire_path,
            robot,
        ):
            shortest_path_calls.append(
                {
                    "floor": floor,
                    "source": np.asarray(source).copy(),
                    "target": np.asarray(target).copy(),
                    "entire_path": entire_path,
                    "robot": robot,
                }
            )
            target = np.asarray(target, dtype=np.float64)
            path = np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 0.35],
                    [float(target[0]), float(target[1])],
                ],
                dtype=np.float64,
            )
            segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            return path, float(np.sum(segment_lengths))

    q_reference = np.arange(28, dtype=np.float64) * 0.01
    q_reference[:6] = 0.0
    robot = SimpleNamespace(
        base_idx=list(range(6)),
        base_control_idx=[0, 1, 5],
        get_joint_positions=lambda: q_reference.copy(),
    )
    robot.scene = SimpleNamespace(trav_map=TraversabilityMap())
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._robot = robot
    backend._compute_base_plan = lambda **_kwargs: pytest.fail(
        "navigate_to must not replace the official full A* path with CuRobo BASE"
    )
    backend._certify_base_trajectory = (
        lambda q, **_kwargs: (
            np.asarray(q, dtype=np.float32),
            {
                "collision_admission": {
                    "available": True,
                    "admitted": True,
                    "world_collision_check": True,
                    "self_collision_check": True,
                    "post_interpolation_check": True,
                },
                "base_trajectory_certificate": {"schema_version": 1},
            },
            {"left": None, "right": None},
        )
    )

    result = backend.plan_navigation_trajectory(
        target_xyz=[2.0, 0.0, 0.0],
        standoff_m=0.6,
        max_travel_m=0.5,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    assert shortest_path_calls
    assert all(call["entire_path"] is True for call in shortest_path_calls)
    assert all(call["robot"] is robot for call in shortest_path_calls)
    q_path = np.asarray(result["joint_trajectory"])
    assert q_path.ndim == 2
    assert q_path.shape[1] == len(q_reference)
    assert np.max(np.abs(q_path[:, 1])) > 0.0
    np.testing.assert_allclose(
        q_path[:, 2:5],
        np.broadcast_to(q_reference[2:5], q_path[:, 2:5].shape),
    )
    np.testing.assert_allclose(
        q_path[:, 6:],
        np.broadcast_to(q_reference[6:], q_path[:, 6:].shape),
    )
    metrics = result["metrics"]["navigation_path"]
    assert metrics["source"] == "official_robot_eroded_traversability"
    assert metrics["entire_path_requested"] is True
    assert metrics["dynamic_world_collision_admission"] is True
    assert metrics["bounded_stage"]["max_travel_m"] == 0.5
    assert metrics["bounded_stage"]["planned_travel_m"] <= 0.5 + 1e-9
    assert metrics["bounded_stage"]["truncated"] is True


class _RelativeTraversabilityMap:
    floor_map = [np.ones((100, 100), dtype=np.uint8)]
    floor_heights = [0.0]

    def __init__(self, *, blocked_cell=None):
        self.blocked_cell = blocked_cell

    @staticmethod
    def world_to_map(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.asarray(
            [
                50 + round((float(xy[1]) - 2.0) * 10.0),
                50 + round((float(xy[0]) - 1.0) * 10.0),
            ],
            dtype=np.int64,
        )

    def _erode_trav_map(self, floor_map, *, robot):
        del robot
        result = floor_map.clone()
        if self.blocked_cell is not None:
            result[self.blocked_cell] = 0
        return result


@pytest.mark.parametrize(
    ("relative_motion", "expected_goal"),
    [
        (
            {"kind": "translation", "direction": "forward", "distance_m": 0.4},
            [1.0, 2.4, math.pi / 2.0],
        ),
        (
            {"kind": "translation", "direction": "backward", "distance_m": 0.4},
            [1.0, 1.6, math.pi / 2.0],
        ),
        (
            {"kind": "rotation", "direction": "left", "angle_deg": 90.0},
            [1.0, 2.0, -math.pi],
        ),
        (
            {"kind": "rotation", "direction": "right", "angle_deg": 90.0},
            [1.0, 2.0, 0.0],
        ),
    ],
)
def test_relative_navigation_plans_body_axis_motion_and_locks_nonbase_q(
    relative_motion,
    expected_goal,
    tmp_path,
):
    q_reference = np.arange(28, dtype=np.float64) * 0.01
    q_reference[:6] = 0.0
    q_reference[0] = 1.0
    q_reference[1] = 2.0
    q_reference[5] = math.pi / 2.0
    robot = SimpleNamespace(
        base_idx=list(range(6)),
        base_control_idx=[0, 1, 5],
        get_joint_positions=lambda: q_reference.copy(),
    )
    robot.scene = SimpleNamespace(trav_map=_RelativeTraversabilityMap())
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._robot = robot
    backend._record_base_phase = lambda _record: None

    def compute_exact_base(*, target_xyyaw, **_kwargs):
        q = q_reference.copy()
        q[0], q[1], q[5] = np.asarray(target_xyyaw, dtype=np.float64)
        return {
            "ok": True,
            "joint_trajectory": q.reshape(1, -1).astype(np.float32),
            "base_goal": np.asarray(target_xyyaw, dtype=np.float64),
            "expected_attachments_by_hand": {"left": None, "right": None},
            "metrics": {
                "collision_admission": {
                    "available": True,
                    "admitted": True,
                    "world_collision_check": True,
                    "self_collision_check": True,
                    "post_interpolation_check": True,
                },
                "base_trajectory_certificate": {"schema_version": 1},
            },
        }

    backend._compute_base_plan = compute_exact_base

    result = backend.plan_relative_navigation_trajectory(
        relative_motion=relative_motion,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    np.testing.assert_allclose(result["base_goal"], expected_goal, atol=1e-6)
    q_path = np.asarray(result["joint_trajectory"])
    nonbase = [index for index in range(28) if index not in {0, 1, 5}]
    np.testing.assert_allclose(
        q_path[:, nonbase],
        np.broadcast_to(q_reference[nonbase], q_path[:, nonbase].shape),
    )
    if relative_motion["kind"] == "translation":
        np.testing.assert_allclose(q_path[:, 5], q_reference[5])
    else:
        np.testing.assert_allclose(
            q_path[:, :2],
            np.broadcast_to(q_reference[:2], q_path[:, :2].shape),
        )
        yaw_steps = np.diff(np.concatenate([[q_reference[5]], q_path[:, 5]]))
        expected_sign = 1.0 if relative_motion["direction"] == "left" else -1.0
        assert np.all(yaw_steps * expected_sign > 0.0)
    path_metrics = result["metrics"]["navigation_path"]
    assert path_metrics["source"] == "official_robot_eroded_traversability"
    assert path_metrics["straight_relative_motion"] is True
    assert path_metrics["dynamic_world_collision_admission"] is True


def test_relative_navigation_rejects_blocked_straight_segment_without_action_plan():
    q_reference = np.zeros(28, dtype=np.float64)
    q_reference[0] = 1.0
    q_reference[1] = 2.0
    q_reference[5] = math.pi / 2.0
    robot = SimpleNamespace(
        base_idx=list(range(6)),
        base_control_idx=[0, 1, 5],
        get_joint_positions=lambda: q_reference.copy(),
    )
    robot.scene = SimpleNamespace(
        trav_map=_RelativeTraversabilityMap(blocked_cell=(52, 50))
    )
    backend = RealCuroboBackend(None)
    backend._robot = robot

    result = backend.plan_relative_navigation_trajectory(
        relative_motion={
            "kind": "translation",
            "direction": "forward",
            "distance_m": 0.4,
        },
        timeout_s=5.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "relative_navigation_untraversable"
    assert "joint_trajectory" not in result


class _NavigationBackend(_FakeBackend):
    def __init__(self, isolation_reports=None):
        super().__init__()
        self.isolation_reports = list(isolation_reports or [])
        self.navigation_actions: list[np.ndarray] = []
        self.relative_navigation_calls: list[dict] = []
        self.navigation_reference = {
            "mode": "base_only",
            "attachments": {
                "left": {"left_eef_link": object()},
                "right": {"right_eef_link": object()},
            },
        }
        self.q_path = np.zeros((3, 28), dtype=np.float32)
        self.q_path[:, 0] = [0.1, 0.2, 0.3]
        self.q_path[:, 1] = [0.0, 0.1, 0.2]
        self.q_path[:, 5] = [0.0, 0.1, 0.2]

    def _navigation_plan(self):
        certificate = {
            "schema_version": 1,
            "trajectory_sha256": hashlib.sha256(
                np.ascontiguousarray(self.q_path, dtype=np.float32).tobytes()
            ).hexdigest(),
            "start_q_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    self.joint_positions, dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "waypoint_count": int(len(self.q_path)),
            "world_collision_check": True,
            "self_collision_check": True,
            "post_interpolation_check": True,
            "attachment_hand_count": 2,
            "colliding_waypoint_count": 0,
        }
        return {
            "ok": True,
            "joint_trajectory": self.q_path.copy(),
            "base_goal": [0.3, 0.2, 0.2],
            "expected_attachments_by_hand": {"left": None, "right": None},
            "metrics": {
                "base_trajectory_certificate": certificate,
                "collision_admission": {
                    "available": True,
                    "admitted": True,
                    "world_collision_check": True,
                    "self_collision_check": True,
                    "post_interpolation_check": True,
                },
                "navigation_path": {
                    "source": "official_robot_eroded_traversability",
                    "dynamic_world_collision_admission": True,
                }
            },
        }

    def plan_navigation_trajectory(self, **_kwargs):
        return self._navigation_plan()

    def plan_relative_navigation_trajectory(self, **kwargs):
        self.relative_navigation_calls.append(kwargs)
        return self._navigation_plan()

    @staticmethod
    def capture_trajectory_hold_reference(*, hand):
        assert hand is None
        return {"mode": "base_only", "token": "fixed"}

    def capture_navigation_isolation_reference(self):
        return self.navigation_reference

    def joint_target_to_action(self, target_q, *, hand, fixed_reference):
        assert hand is None
        assert fixed_reference == {"mode": "base_only", "token": "fixed"}
        target_q = np.asarray(target_q)
        action = self.hold.copy()
        action[ENV_ACTION_SEGMENTS["base"]] = target_q[[0, 1, 5]]
        self.base_pose = target_q[[0, 1, 5]].astype(np.float64)
        self.navigation_actions.append(action.copy())
        return action

    def navigation_isolation_report(self, *, action, reference):
        assert reference is self.navigation_reference
        assert np.asarray(action).shape == (23,)
        if self.isolation_reports:
            return self.isolation_reports.pop(0)
        return {
            "available": True,
            "ok": True,
            "mode": "base_only",
            "checks": {
                "base_z_locked": True,
                "base_roll_pitch_locked": True,
                "trunk_locked": True,
                "left_arm_locked": True,
                "right_arm_locked": True,
                "left_gripper_command_locked": True,
                "right_gripper_command_locked": True,
                "left_attachment_identity_unchanged": True,
                "right_attachment_identity_unchanged": True,
            },
            "max_observed": {},
        }

    @staticmethod
    def joint_tracking_report(target_q, *, hand):
        del target_q
        assert hand is None
        return {
            "available": True,
            "reached": True,
            "max_base_xy_error_m": 0.0,
            "base_yaw_error_rad": 0.0,
        }


@pytest.mark.parametrize(
    ("action", "expected_motion"),
    [
        (
            "forward",
            {"kind": "translation", "direction": "forward", "distance_m": 0.05},
        ),
        (
            "backward",
            {"kind": "translation", "direction": "backward", "distance_m": 0.05},
        ),
        (
            "turn_left",
            {"kind": "rotation", "direction": "left", "angle_deg": 5.0},
        ),
        (
            "turn_right",
            {"kind": "rotation", "direction": "right", "angle_deg": 5.0},
        ),
    ],
)
def test_jog_base_uses_only_fixed_server_steps(action, expected_motion):
    backend = _NavigationBackend()
    executor, _env = _executor(backend)

    result = executor.jog_base(action, timeout_s=5.0)

    assert result["primitive_success"] is True
    assert backend.relative_navigation_calls == [
        {"relative_motion": expected_motion, "timeout_s": 5.0}
    ]
    assert result["metrics"]["fixed_server_step"] is True


def test_navigation_missing_collision_certificate_executes_zero_actions():
    backend = _NavigationBackend()
    unsafe_plan = backend._navigation_plan()
    unsafe_plan["metrics"].pop("base_trajectory_certificate")
    backend.plan_navigation_trajectory = lambda **_kwargs: unsafe_plan
    executor, env = _executor(backend)

    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_collision_certificate_unavailable"
    assert result["metrics"]["env_actions_sent"] == 0
    assert env.calls == []


def test_jog_eef_transforms_base_local_delta_and_preserves_call_start_quat():
    backend = _FakeBackend()
    backend.base_pose[2] = math.pi / 2.0
    backend.quat = np.array([0.0, 0.0, math.sin(0.2), math.cos(0.2)])
    call_start_quat = backend.quat.copy()
    executor, _env = _executor(backend)

    result = executor.jog_eef("left", "forward", timeout_s=10.0)

    assert result["primitive_success"] is True
    np.testing.assert_allclose(
        backend.planned_targets[0],
        [0.5, 0.03, 0.0],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        backend.whole_body_plan_calls[0]["target_quat_xyzw"],
        call_start_quat,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result["metrics"]["requested_delta_world"],
        [0.0, 0.03, 0.0],
        atol=1e-8,
    )


def test_jog_eef_fallback_is_plan_only_single_axis_and_bounded():
    class FallbackBackend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            target = np.asarray(kwargs["target_xyz"], dtype=np.float64)
            self.planned_targets.append(target.copy())
            if np.allclose(target, [0.53, 0.0, 0.0], atol=1e-9):
                return {
                    "ok": False,
                    "stop_reason": "unreachable",
                    "metrics": {"env_actions_sent": 0},
                }
            # Avoid recording the same target twice in the base fake.
            self.planned_targets.pop()
            return super().plan_whole_body_trajectory(**kwargs)

    backend = FallbackBackend()
    executor, env = _executor(backend)

    result = executor.jog_eef("right", "forward", timeout_s=10.0)

    assert result["primitive_success"] is True
    attempts = result["metrics"]["candidate_attempts"]
    assert attempts[0]["fallback_offset"] == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(
        attempts[1]["fallback_offset"],
        [0.0, 0.0025, 0.0],
        atol=1e-9,
    )
    assert np.count_nonzero(np.abs(attempts[1]["fallback_offset"]) > 0.0) == 1
    assert max(abs(value) for value in attempts[1]["fallback_offset"]) <= 0.005
    assert len(env.calls) > 0


def test_jog_eef_non_unreachable_failure_does_not_try_compensation():
    class FailedBackend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            self.whole_body_plan_calls.append(dict(kwargs))
            return {
                "ok": False,
                "stop_reason": "planner_unavailable",
                "metrics": {"env_actions_sent": 0},
            }

    backend = FailedBackend()
    executor, env = _executor(backend)

    result = executor.jog_eef("left", "up", timeout_s=10.0)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "planner_unavailable"
    assert len(backend.whole_body_plan_calls) == 1
    assert len(result["metrics"]["candidate_attempts"]) == 1
    assert env.calls == []


def test_jog_torso_fails_closed_without_verified_curobo_controller():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    result = executor.jog_torso("up")

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "torso_control_unsupported"
    assert result["metrics"]["target_link"] == "torso_link4"
    assert result["metrics"]["requested_delta_z_m"] == pytest.approx(0.03)
    assert result["metrics"]["env_actions_sent"] == 0
    assert env.calls == []


def _add_wrist_frame(executor, hand):
    intrinsics = CameraIntrinsics(
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
        width=5,
        height=5,
    )
    executor.frame_cache.add(
        camera=f"{hand}_wrist",
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=np.ones((5, 5), dtype=np.float32),
        intrinsics=intrinsics,
        camera_to_world=np.eye(4),
        step_index=1,
        frame_id=f"{hand}-wrist-1",
    )


def test_jog_wrist_is_unavailable_before_real_visual_probe():
    backend = _FakeBackend()
    executor, env = _executor(backend)
    _add_wrist_frame(executor, "left")

    result = executor.jog_wrist("left", "rotate_left", timeout_s=10.0)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "wrist_calibration_unavailable"
    assert result["metrics"]["env_actions_sent"] == 0
    assert env.calls == []


@pytest.mark.parametrize(("hand", "sign"), [("left", 1.0), ("right", -1.0)])
def test_jog_wrist_uses_independent_verified_visual_sign_and_holds_position(
    hand,
    sign,
):
    class CalibratedBackend(_FakeBackend):
        @staticmethod
        def wrist_visual_rotation_capability(selected_hand):
            return {
                "verified": True,
                "hand": selected_hand,
                "clockwise_angle_sign": (
                    1.0 if selected_hand == "left" else -1.0
                ),
                "probe_artifact": f"{selected_hand}-visual-probe.json",
            }

    backend = CalibratedBackend()
    start_position = backend.pose.copy()
    executor, env = _executor(backend)
    _add_wrist_frame(executor, hand)

    result = executor.jog_wrist(hand, "rotate_left", timeout_s=10.0)

    assert result["primitive_success"] is True
    assert result["metrics"]["requested_rotation_rad"] == pytest.approx(
        sign * math.radians(5.0)
    )
    assert result["metrics"]["final_position_drift_m"] <= 0.005
    np.testing.assert_allclose(backend.pose, start_position, atol=1e-9)
    assert len(env.calls) > 0


@pytest.mark.parametrize("hand", ["left", "right"])
def test_set_gripper_is_formal_latched_primitive_without_retreat(hand):
    backend = _FakeBackend()
    executor, env = _executor(backend)

    result = executor.set_gripper(hand, 0.0, timeout_s=10.0)

    assert result["primitive_success"] is True
    assert result["metrics"]["manual_primitive"] == "set_gripper"
    assert result["metrics"]["retreat_executed"] is False
    assert result["metrics"]["network_primitive_calls"] == 1
    assert env._gripper_latch[hand] == -1.0
    assert len(env.calls) > 0


def test_real_backend_default_artifacts_never_use_repository_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = RealCuroboBackend(None)

    backend._record_base_phase({"phase": "test"})

    assert backend.output_dir != tmp_path
    assert not (tmp_path / "planner_base_phases.jsonl").exists()
    assert (backend.output_dir / "planner_base_phases.jsonl").is_file()


def test_navigation_execution_emits_base_only_actions_and_preserves_attachments():
    backend = _NavigationBackend()
    executor, env = _executor(backend)

    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.calls) == len(backend.q_path) == 3
    assert result["metrics"]["navigation_isolation"]["ok"] is True
    checks = result["metrics"]["navigation_isolation"]["checks"]
    assert checks["left_attachment_identity_unchanged"] is True
    assert checks["right_attachment_identity_unchanged"] is True
    for action in backend.navigation_actions:
        for segment_name in (
            "trunk",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        ):
            segment = ENV_ACTION_SEGMENTS[segment_name]
            np.testing.assert_allclose(action[segment], backend.hold[segment])


def test_navigation_executor_dispatches_relative_motion_to_relative_planner():
    backend = _NavigationBackend()
    executor, env = _executor(backend)
    motion = {
        "kind": "rotation",
        "direction": "left",
        "angle_deg": 90.0,
    }

    result = executor.navigate_to(relative_motion=motion, timeout_s=5.0)

    assert result["primitive_success"] is True
    assert backend.relative_navigation_calls == [
        {
            "relative_motion": motion,
            "timeout_s": 5.0,
        }
    ]
    assert len(env.calls) == len(backend.q_path)


@pytest.mark.parametrize(
    ("second_report", "expected_stop"),
    [
        (
            {
                "available": False,
                "ok": False,
                "mode": "base_only",
                "checks": {},
                "max_observed": {},
                "reason": "feedback missing",
            },
            "navigation_isolation_feedback_unavailable",
        ),
        (
            {
                "available": True,
                "ok": False,
                "mode": "base_only",
                "checks": {
                    "left_attachment_identity_unchanged": True,
                    "right_attachment_identity_unchanged": False,
                },
                "max_observed": {},
            },
            "navigation_isolation_violation",
        ),
    ],
)
def test_navigation_isolation_failure_stops_before_the_next_action(
    second_report,
    expected_stop,
):
    first_report = {
        "available": True,
        "ok": True,
        "mode": "base_only",
        "checks": {
            "left_attachment_identity_unchanged": True,
            "right_attachment_identity_unchanged": True,
        },
        "max_observed": {},
    }
    backend = _NavigationBackend([first_report, second_report])
    executor, env = _executor(backend)

    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == expected_stop
    assert len(env.calls) == 2
    assert len(backend.navigation_actions) == 2
    assert result["metrics"]["executed_waypoints"] == 2


@pytest.mark.parametrize(
    ("raw_success", "terminated", "truncated", "expected_success", "expected_stop"),
    [
        (True, False, False, True, "official_task_success"),
        (False, True, False, False, "environment_terminated"),
        (False, False, True, False, "environment_truncated"),
    ],
)
def test_navigation_hard_runtime_boundaries_stop_after_exact_executed_step(
    raw_success,
    terminated,
    truncated,
    expected_success,
    expected_stop,
):
    backend = _NavigationBackend()
    executor, env = _executor(backend)

    def one_boundary_step(actions):
        env.calls.append(np.asarray(actions).copy())
        return (
            None,
            0.0,
            terminated,
            truncated,
            {
                "done": {"success": raw_success},
                "_rpent": {"executed_steps": 1},
            },
        )

    env.chunk_step = one_boundary_step
    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is expected_success
    assert result["task_success"] is raw_success
    assert result["stop_reason"] == expected_stop
    assert len(env.calls) == 1
    assert len(backend.navigation_actions) == 1


def test_pi0_nav_pick_schema_remains_outside_trunk_assist_contract():
    properties = PI0_NAV_PICK_SPEC["input_schema"]["properties"]

    assert "allow_trunk_assist" not in properties
    assert "motion_scope" not in properties
    assert "role" not in properties
    assert "visual_hand_check" not in properties
    assert "[32,23]" in PI0_NAV_PICK_SPEC["description"]


def _real_isolation_fixture(*, selected_hand="left", gripper_only=False):
    q = np.zeros(28, dtype=np.float64)
    robot = SimpleNamespace(
        base_idx=np.arange(0, 6),
        trunk_control_idx=np.arange(6, 10),
        arm_control_idx={
            "left": np.arange(10, 17),
            "right": np.arange(17, 24),
        },
        _ag_obj_in_hand={"left": None, "right": None},
        get_joint_positions=lambda: q.copy(),
    )
    eef_poses = {
        "left": [
            np.array([0.4, 0.2, 0.8], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ],
        "right": [
            np.array([0.4, -0.2, 0.8], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ],
    }
    backend = RealCuroboBackend(None)
    backend.env_facade = SimpleNamespace(_gripper_latch={"left": 0.0, "right": 0.0})
    backend._find_robot = lambda: robot
    backend.get_eef_pose = lambda hand: tuple(
        np.asarray(value).copy() for value in eef_poses[hand]
    )
    reference = backend.capture_single_arm_isolation_reference(
        hand=selected_hand,
        gripper_only=gripper_only,
    )
    reference["context_id"] = "test-isolation-context"
    action = np.zeros(23, dtype=np.float32)
    return backend, robot, q, eef_poses, reference, action


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("gripper_only", [False, True])
def test_real_single_arm_isolation_report_has_complete_public_shape(
    hand,
    gripper_only,
):
    backend, _robot, _q, _eef, reference, action = _real_isolation_fixture(
        selected_hand=hand,
        gripper_only=gripper_only,
    )

    report = backend.single_arm_isolation_report(
        hand=hand,
        action=action,
        reference=reference,
        gripper_only=gripper_only,
    )

    assert report["available"] is True
    assert report["ok"] is True
    assert report["selected_hand"] == hand
    assert report["mode"] == ("gripper_only" if gripper_only else "arm_motion")
    assert report["context_id"] == "test-isolation-context"
    expected_checks = {
        "locked_joints",
        "inactive_eef",
        "locked_gripper_commands",
        "inactive_attachment",
    }
    if not gripper_only:
        expected_checks.add("selected_attachment")
    assert set(report["checks"]) == expected_checks
    assert all(check["ok"] is True for check in report["checks"].values())
    assert report["thresholds"]["base_xy_m"] == pytest.approx(0.01)
    assert report["thresholds"]["base_z_m"] == pytest.approx(0.01)
    assert report["thresholds"]["base_rpy_rad"] == pytest.approx(np.deg2rad(1.0))
    assert report["thresholds"]["articulation_rad"] == pytest.approx(0.01)
    assert report["thresholds"]["inactive_eef_position_m"] == pytest.approx(0.01)
    assert report["thresholds"]["inactive_eef_orientation_rad"] == pytest.approx(
        np.deg2rad(1.0)
    )
    assert report["thresholds"]["gripper_command"] == pytest.approx(1e-6)
    assert "prim_path" not in repr(report)


@pytest.mark.parametrize(
    ("failure", "failed_check"),
    [
        ("base", "locked_joints"),
        ("trunk", "locked_joints"),
        ("inactive_arm", "locked_joints"),
        ("inactive_eef", "inactive_eef"),
        ("inactive_gripper", "locked_gripper_commands"),
        ("inactive_attachment", "inactive_attachment"),
    ],
)
def test_real_single_arm_isolation_report_detects_each_locked_scope(
    failure,
    failed_check,
):
    backend, robot, q, eef_poses, reference, action = _real_isolation_fixture(
        selected_hand="left",
        gripper_only=False,
    )
    if failure == "base":
        q[0] = 0.011
    elif failure == "trunk":
        q[6] = 0.011
    elif failure == "inactive_arm":
        q[17] = 0.011
    elif failure == "inactive_eef":
        eef_poses["right"][0][0] += 0.011
    elif failure == "inactive_gripper":
        action[ENV_ACTION_SEGMENTS["right_gripper"]] = 2e-6
    elif failure == "inactive_attachment":
        robot._ag_obj_in_hand["right"] = SimpleNamespace(root_link=object())

    report = backend.single_arm_isolation_report(
        hand="left",
        action=action,
        reference=reference,
        gripper_only=False,
    )

    assert report["available"] is True
    assert report["ok"] is False
    assert report["checks"][failed_check]["ok"] is False
    assert "prim_path" not in repr(report)


@pytest.mark.parametrize("transition", ["none_to_attached", "identity_replaced"])
def test_arm_motion_report_rejects_selected_attachment_transition(transition):
    backend, robot, _q, _eef_poses, reference, action = _real_isolation_fixture(
        selected_hand="left",
        gripper_only=False,
    )
    original_root = object()
    replacement_root = object()
    if transition == "identity_replaced":
        robot._ag_obj_in_hand["left"] = SimpleNamespace(root_link=original_root)
        reference = backend.capture_single_arm_isolation_reference(
            hand="left",
            gripper_only=False,
        )
        reference["context_id"] = "selected-attachment-transition"
    robot._ag_obj_in_hand["left"] = SimpleNamespace(root_link=replacement_root)

    report = backend.single_arm_isolation_report(
        hand="left",
        action=action,
        reference=reference,
        gripper_only=False,
    )

    assert report["available"] is True
    assert report["ok"] is False
    assert report["checks"]["selected_attachment"]["ok"] is False
    assert report["checks"]["selected_attachment"]["matches"] is False
    assert "prim_path" not in repr(report)


@pytest.mark.parametrize("transition", ["none_to_attached", "identity_replaced"])
def test_arm_motion_selected_attachment_fault_stops_before_next_action(
    transition,
):
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attachments = {"left": None, "right": None}
            self.original_root = object()
            self.replacement_root = object()
            if transition == "identity_replaced":
                self.attachments["left"] = {
                    "left_eef_link": self.original_root,
                }

        def get_attached_object(self, hand):
            return self.attachments[hand]

        def advance(self):
            if len(self.env.calls) == 2:
                self.attachments["left"] = {
                    "left_eef_link": self.replacement_root,
                }

    backend = Backend()
    executor, env = _executor(backend)

    result = executor._execute_actions(
        np.zeros((4, 23), dtype=np.float32),
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "single_arm_isolation_violation"
    assert len(env.calls) == 2
    assert result["metrics"]["executed_waypoints"] == 2
    selected_check = result["metrics"]["single_arm_isolation"]["checks"][
        "selected_attachment"
    ]
    assert selected_check["ok"] is False
    assert selected_check["matches"] is False


def test_gripper_only_allows_selected_attachment_transition():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attachments = {"left": None, "right": None}
            self.selected_root = object()

        def get_attached_object(self, hand):
            return self.attachments[hand]

        def advance(self):
            if len(self.env.calls) == 1:
                self.attachments["left"] = {
                    "left_eef_link": self.selected_root,
                }

    backend = Backend()
    executor, env = _executor(backend)

    result = executor._execute_actions(
        np.zeros((2, 23), dtype=np.float32),
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.0,
        orientation_tolerance_rad=0.0,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        static_gripper_only=True,
    )

    assert result["primitive_success"] is True
    assert len(env.calls) == 2
    isolation = result["metrics"]["single_arm_isolation"]
    assert isolation["ok"] is True
    assert "selected_attachment" not in isolation["checks"]


def test_gripper_only_still_rejects_inactive_attachment_identity_drift():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.original_root = object()
            self.replacement_root = object()
            self.attachments = {
                "left": None,
                "right": {"right_eef_link": self.original_root},
            }

        def get_attached_object(self, hand):
            return self.attachments[hand]

        def advance(self):
            if len(self.env.calls) == 1:
                self.attachments["right"] = {
                    "right_eef_link": self.replacement_root,
                }

    backend = Backend()
    executor, env = _executor(backend)

    result = executor._execute_actions(
        np.zeros((3, 23), dtype=np.float32),
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.0,
        orientation_tolerance_rad=0.0,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
        static_gripper_only=True,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "single_arm_isolation_violation"
    assert len(env.calls) == 1
    isolation = result["metrics"]["single_arm_isolation"]
    assert "selected_attachment" not in isolation["checks"]
    assert isolation["checks"]["inactive_attachment"]["ok"] is False


@pytest.mark.parametrize(
    ("available", "expected_reason"),
    [
        (False, "single_arm_isolation_feedback_unavailable"),
        (True, "single_arm_isolation_violation"),
    ],
)
def test_isolation_fault_stops_exactly_before_the_next_action(
    available,
    expected_reason,
):
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.report_count = 0

        def single_arm_isolation_report(self, **kwargs):
            self.report_count += 1
            report = super().single_arm_isolation_report(**kwargs)
            if self.report_count == 2:
                report["available"] = available
                report["ok"] = False
                report["checks"]["inactive_eef"]["ok"] = False
                report["max_observed"]["inactive_eef_position_m"] = 0.011
                if not available:
                    report["reason"] = "injected feedback loss"
            return report

    backend = Backend()
    executor, env = _executor(backend)

    result = executor._execute_actions(
        np.zeros((4, 23), dtype=np.float32),
        hand="left",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == expected_reason
    assert backend.report_count == 2
    assert len(env.calls) == 2
    assert result["metrics"]["executed_waypoints"] == 2
    isolation = result["metrics"]["single_arm_isolation"]
    assert isolation["ok"] is False
    assert isolation["checks_performed"] == 2


def test_missing_isolation_reference_fails_before_any_action():
    backend = _FakeBackend()
    backend.capture_single_arm_isolation_reference = lambda **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("injected missing feedback"))
    executor, env = _executor(backend)

    result = executor._execute_actions(
        np.zeros((2, 23), dtype=np.float32),
        hand="right",
        target_xyz=None,
        target_quat_xyzw=None,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=False,
        hold_steps_required=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "single_arm_isolation_feedback_unavailable"
    assert result["metrics"]["executed_waypoints"] == 0
    assert env.calls == []


def test_press_uses_whole_body_without_single_arm_isolation_context():
    backend = _FakeBackend(contact_mode="expected_near_target")
    executor, _ = _executor(backend)

    result = executor.press(
        hand="right",
        target_xyz=[0.5, 0.0, -0.1],
        press_direction=[0.0, 0.0, -1.0],
        travel_m=0.004,
    )

    assert result["primitive_success"] is True
    assert result["metrics"]["motion_scope"] == "whole_body"
    assert result["metrics"]["precontact_motion"]["motion_scope"] == "whole_body"
    assert result["metrics"]["single_arm_isolation"] is None
    assert result["metrics"]["precontact_motion"].get("single_arm_isolation") is None
    assert backend.isolation_capture_calls == []


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


def test_q_trajectory_does_not_query_locked_joint_drift_reporter():
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
            raise AssertionError("locked-joint drift telemetry must not gate execution")

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

    assert result["primitive_success"] is True
    assert len(env.calls) == 3
    assert backend.drift_calls == 0


def test_q_trajectory_does_not_query_joint_margin_reporter():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.margin_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            del target_q, hand
            return np.zeros(23, dtype=np.float32)

        def joint_margin_report(self):
            self.margin_calls += 1
            raise AssertionError("joint-margin telemetry must not gate execution")

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

    assert result["primitive_success"] is True
    assert len(env.calls) == 1
    assert backend.margin_calls == 0


def test_q_trajectory_does_not_query_runtime_collision_reporter():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.collision_forces = []

        def joint_target_to_action(self, target_q, *, hand):
            return np.zeros(23, dtype=np.float32)

        def collision_report(self, *, force=False):
            self.collision_forces.append(force)
            raise AssertionError("collision telemetry must not gate execution")

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
        joint_trajectory=np.zeros((4, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is True
    assert len(env.calls) == 4
    assert backend.collision_forces == []


def test_q_trajectory_does_not_query_actual_dynamics_reporter():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.dynamics_calls = 0

        def joint_target_to_action(self, target_q, *, hand):
            del target_q, hand
            return np.zeros(23, dtype=np.float32)

        def dynamics_report(self):
            self.dynamics_calls += 1
            raise AssertionError("actual-dynamics telemetry must not gate execution")

    backend = Backend()
    executor, _ = _executor(backend)
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
    assert result["stop_reason"] == "reached"
    assert backend.dynamics_calls == 0


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
    assert result["stop_reason"] == "single_arm_isolation_violation"
    assert len(env.calls) == 3
    assert (
        result["metrics"]["single_arm_isolation"]["checks"]["selected_attachment"]["ok"]
        is False
    )


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
        "elapsed_s",
        "trace",
        "trace_artifact",
    ):
        assert key in result


def _load_base_curobo_probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_base_curobo_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rpent_behavior_base_curobo_probe_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_curobo_probe_resolves_public_seed_from_task_spec():
    probe = _load_base_curobo_probe_module()

    radio, radio_seed = probe._resolve_probe_identity("turning_on_radio", 211)
    trash, trash_seed = probe._resolve_probe_identity("picking_up_trash", 108)

    assert radio.task_index == 0
    assert radio_seed == 6
    assert radio.tag(radio_seed) == "turning_on_radio_s6"
    assert trash.task_index == 1
    assert trash_seed == 10
    with pytest.raises(ValueError, match="not a mapped public instance"):
        probe._resolve_probe_identity("turning_on_radio", 999_999)


def _probe_test_args(tmp_path):
    return SimpleNamespace(
        activity_instance_dir=str(
            tmp_path / "house_double_floor_lower_task_turning_on_radio_instances"
        ),
        output_dir=str(tmp_path / "probe"),
        task_name="turning_on_radio",
        seed=0,
        activity_instance_id=211,
        compatible_only=False,
        execute_sim_jogs=False,
        probe_arm_candidates=False,
    )


def test_base_curobo_probe_config_failure_writes_structured_report(
    tmp_path,
    monkeypatch,
):
    probe = _load_base_curobo_probe_module()

    monkeypatch.setattr(probe, "_args", lambda: _probe_test_args(tmp_path))

    def fail_config(_args):
        raise FileNotFoundError("synthetic config failure")

    monkeypatch.setattr(probe, "_load_env_config", fail_config)
    with pytest.raises(FileNotFoundError, match="synthetic config failure"):
        probe.main()

    output_dir = tmp_path / "probe"
    report = json.loads((output_dir / "base_curobo_probe.json").read_text())
    events = [
        json.loads(line)
        for line in (
            output_dir / "base_curobo_probe.events.jsonl"
        ).read_text().splitlines()
    ]
    assert report["status"] == "failed"
    assert report["failed_stage"] == "load_env_config"
    assert report["configuration"]["public_seed"] == 6
    assert report["configuration"]["attempt_index"] == 1
    assert report["configuration"]["seed"] == 0
    failure_event = next(
        event for event in events if event["event"] == "probe_failed"
    )
    assert failure_event["stage"] == "load_env_config"
    assert events[-1]["event"] == "probe_lifecycle_sealed"


def test_base_curobo_probe_facade_failure_preserves_complete_identity(
    tmp_path,
    monkeypatch,
    capsys,
):
    probe = _load_base_curobo_probe_module()

    args = _probe_test_args(tmp_path)
    captured = {}
    monkeypatch.setattr(probe, "_args", lambda: args)
    monkeypatch.setattr(probe, "_load_env_config", lambda _args: object())

    def fail_facade(*, cfg, meta, output_dir):
        captured.update(cfg=cfg, meta=meta, output_dir=output_dir)
        raise RuntimeError("synthetic facade failure")

    monkeypatch.setattr(probe, "BehaviorEnvFacade", fail_facade)
    with pytest.raises(RuntimeError, match="synthetic facade failure"):
        probe.main()

    report = json.loads(
        (tmp_path / "probe" / "base_curobo_probe.json").read_text()
    )
    assert captured["meta"]["public_seed"] == 6
    assert captured["meta"]["attempt_index"] == 1
    assert captured["meta"]["activity_instance_id"] == 211
    assert report["failed_stage"] == "construct_facade"
    stderr = capsys.readouterr().err
    assert '"event": "probe_failed"' in stderr
    assert '"stage": "construct_facade"' in stderr


def test_base_curobo_probe_failure_uses_shutdown_not_gripper_close(
    tmp_path,
    monkeypatch,
):
    probe = _load_base_curobo_probe_module()
    args = _probe_test_args(tmp_path)
    calls = {"shutdown": 0, "close": 0, "flush": 0}

    class Facade:
        _env_steps = 17

        def reset(self):
            raise RuntimeError("synthetic reset failure")

        def shutdown(self):
            calls["shutdown"] += 1

        def close(self, *, hand, visual_hand_check):
            del hand, visual_hand_check
            calls["close"] += 1
            raise AssertionError("gripper close must never be lifecycle cleanup")

    facade = Facade()
    monkeypatch.setattr(probe, "_args", lambda: args)
    monkeypatch.setattr(probe, "_load_env_config", lambda _args: object())
    monkeypatch.setattr(
        probe,
        "BehaviorEnvFacade",
        lambda **_kwargs: facade,
    )
    monkeypatch.setattr(
        probe,
        "_flush_shutdown_artifacts",
        lambda _output_dir: calls.__setitem__("flush", calls["flush"] + 1),
    )

    with pytest.raises(RuntimeError, match="synthetic reset failure"):
        probe.main()

    report = json.loads(
        (tmp_path / "probe" / "base_curobo_probe.json").read_text()
    )
    assert calls == {"shutdown": 1, "close": 0, "flush": 1}
    assert report["cleanup"]["method"] == "BehaviorEnvFacade.shutdown"
    assert report["cleanup"]["gripper_close_called"] is False
    assert report["cleanup"]["env_step_delta"] == 0
    assert report["cleanup"]["artifacts_flushed"] is True
