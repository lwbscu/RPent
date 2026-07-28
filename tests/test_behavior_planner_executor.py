import hashlib
import importlib.util
import inspect
import json
import math
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import CameraIntrinsics, FrameCache
from robots.behavior.planner_executor import (
    BASE_ACTIVE_JOINT_NAMES,
    BASE_EXECUTION_XY_STEP_M,
    BASE_EXECUTION_YAW_STEP_RAD,
    BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD,
    BASE_TERMINAL_POSITION_TOLERANCE_M,
    DASHBOARD_BASE_EXECUTION_XY_STEP_M,
    DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD,
    DASHBOARD_PREPARED_BASE_PLANNING_PROFILE,
    EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD,
    EEF_TERMINAL_POSITION_TOLERANCE_M,
    LOCAL_GUARDED_IK_SEEDS,
    PREPARED_DASHBOARD_BASE_EXECUTION_POLICY,
    PREPARED_DASHBOARD_EEF_EXECUTION_POLICY,
    RESET_IDENTITY_WARMUP_PROFILE,
    RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
    TERMINAL_COMMAND_LIMIT,
    TRACKING_HARD_ARTICULATION_ERROR_RAD,
    TRACKING_HARD_BASE_XY_ERROR_M,
    TRACKING_HARD_BASE_YAW_ERROR_RAD,
    WHOLE_BODY_ACTIVE_JOINT_NAMES,
    WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S,
    WHOLE_BODY_DASHBOARD_JOG_LOCAL_IK_DEADLINE_S,
    WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S,
    WHOLE_BODY_DASHBOARD_JOG_REPLAN_POSITION_IMPROVEMENT_M,
    WHOLE_BODY_DENSE_COLLISION_STEP,
    WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M,
    WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M,
    WHOLE_BODY_EEF_NUMERICAL_MARGIN_M,
    WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M,
    WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M,
    WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M,
    WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD,
    WHOLE_BODY_EXECUTION_BASE_XY_STEP_M,
    WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD,
    WHOLE_BODY_LOCKED_JOINT_NAMES,
    WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M,
    WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
    WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
    PlannerExecutor,
    RealCuroboBackend,
    _apply_single_arm_isolation_mask,
    _attachment_identity_status,
    _canonical_base_xyyaw,
    _canonicalize_whole_body_base_yaw_trajectory,
    _eef_pose_path_admission_report,
    _guarded_waypoint_distances,
    _interpolate_joint_trajectory,
    _interpolate_whole_body_execution_trajectory,
    _minimum_jerk_base_execution_trajectory,
    _quat_to_intrinsic_rpy,
    _retime_joint_trajectory,
    _terminally_smoothed_joint_trajectory,
    _tracking_hard_deviation_report,
    _wall_clock_deadline,
    _whole_body_execution_step_report,
    _whole_body_execution_subset_indices,
    _whole_body_search_profile,
    _whole_body_target_sha256,
    _wrap_angle,
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
    ("field", "limit"),
    [
        ("max_base_xy_error_m", TRACKING_HARD_BASE_XY_ERROR_M),
        ("base_yaw_error_rad", TRACKING_HARD_BASE_YAW_ERROR_RAD),
        (
            "max_articulation_error_rad",
            TRACKING_HARD_ARTICULATION_ERROR_RAD,
        ),
    ],
)
def test_wide_tracking_hard_limits_are_inclusive_and_stop_above(
    field,
    limit,
):
    report = {
        "available": True,
        "max_base_xy_error_m": 0.0,
        "base_yaw_error_rad": 0.0,
        "max_articulation_error_rad": 0.0,
    }
    report[field] = limit
    boundary, available = _tracking_hard_deviation_report(report)
    assert available is True
    assert boundary["hard_deviation"] is False

    report[field] = np.nextafter(limit, math.inf)
    exceeded, available = _tracking_hard_deviation_report(report)
    assert available is True
    assert exceeded["hard_deviation"] is True
    assert exceeded["checks"][{
        "max_base_xy_error_m": "base_xy",
        "base_yaw_error_rad": "base_yaw",
        "max_articulation_error_rad": "articulation",
    }[field]] is False


def test_whole_body_execution_caps_are_sparse_safe_and_inclusive():
    names = tuple(_FakeBackend().joint_names)
    by_name = {name: index for index, name in enumerate(names)}
    q = np.zeros((2, len(names)), dtype=np.float32)
    q[1, by_name["base_footprint_x_joint"]] = (
        WHOLE_BODY_EXECUTION_BASE_XY_STEP_M
    )
    q[1, by_name["torso_joint1"]] = (
        WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
    )
    positions = np.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.5 + WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M, 0.0, 0.0],
        ]
    )
    quaternions = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )

    report = _whole_body_execution_step_report(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )

    assert WHOLE_BODY_EXECUTION_BASE_XY_STEP_M == pytest.approx(
        WHOLE_BODY_DENSE_COLLISION_STEP
    )
    assert WHOLE_BODY_DENSE_COLLISION_STEP == pytest.approx(0.0075)
    assert WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M == pytest.approx(0.0022)
    assert WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M == pytest.approx(0.006)
    assert WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD == pytest.approx(0.03)
    assert WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD < (
        TRACKING_HARD_ARTICULATION_ERROR_RAD
    )
    assert report["admitted"] is True
    assert report["base_xy_step_limit_m"] == pytest.approx(0.0075)
    assert report["articulation_step_limit_rad"] == pytest.approx(0.03)
    assert report["eef_cartesian_step_limit_m"] == pytest.approx(0.006)

    eef_over_limit = positions.copy()
    eef_over_limit[1, 0] += 2e-6
    eef_rejected = _whole_body_execution_step_report(
        q,
        eef_over_limit,
        quaternions,
        joint_names=names,
        short_target=True,
    )
    assert eef_rejected["admitted"] is False
    assert eef_rejected["checks"]["eef_cartesian_step"] is False

    articulation_over_limit = q.copy()
    articulation_over_limit[1, by_name["torso_joint1"]] += np.float32(2e-6)
    articulation_rejected = _whole_body_execution_step_report(
        articulation_over_limit,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )
    assert articulation_rejected["admitted"] is False
    assert articulation_rejected["checks"]["articulation_step"] is False

    q[1, by_name["base_footprint_x_joint"]] += np.float32(2e-6)
    rejected = _whole_body_execution_step_report(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )
    assert rejected["admitted"] is False
    assert rejected["checks"]["base_x_step"] is False


def test_short_eef_execution_subset_uses_five_certified_commands_for_3cm():
    names = tuple(_FakeBackend().joint_names)
    by_name = {name: index for index, name in enumerate(names)}
    sample_count = 31
    q = np.zeros((sample_count, len(names)), dtype=np.float32)
    q[:, by_name["left_arm_joint1"]] = np.linspace(
        0.0,
        0.12,
        sample_count,
        dtype=np.float32,
    )
    positions = np.zeros((sample_count, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.03, sample_count)
    quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        sample_count,
        axis=0,
    )

    indices = _whole_body_execution_subset_indices(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )
    execution_report = _whole_body_execution_step_report(
        q[indices],
        positions[indices],
        quaternions[indices],
        joint_names=names,
        short_target=True,
    )

    np.testing.assert_array_equal(indices, [0, 6, 12, 18, 24, 30])
    assert len(indices) - 1 == 5
    assert execution_report["admitted"] is True
    assert execution_report["max_eef_cartesian_step_m"] == pytest.approx(0.006)
    assert execution_report["max_articulation_step_rad"] == pytest.approx(
        0.024
    )


def _reference_whole_body_execution_subset_indices(
    q,
    positions,
    quaternions,
    *,
    joint_names,
    short_target,
):
    q = np.asarray(q, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    _whole_body_execution_step_report(
        q,
        positions,
        quaternions,
        joint_names=joint_names,
        short_target=short_target,
    )
    edge_count = [None] * len(q)
    predecessor = np.full((len(q),), -1, dtype=np.int64)
    edge_count[0] = 0
    for end in range(1, len(q)):
        best_key = None
        best_start = -1
        for start in range(end):
            if edge_count[start] is None:
                continue
            report = _whole_body_execution_step_report(
                q[[start, end]],
                positions[[start, end]],
                quaternions[[start, end]],
                joint_names=joint_names,
                short_target=short_target,
            )
            if report["admitted"] is not True:
                continue
            key = (int(edge_count[start]) + 1, -start)
            if best_key is None or key < best_key:
                best_key = key
                best_start = start
        if best_key is not None:
            edge_count[end] = best_key[0]
            predecessor[end] = best_start
    if predecessor[-1] < 0:
        raise RuntimeError(
            "whole-body dense path has no execution subset satisfying step limits"
        )
    reverse_indices = [len(q) - 1]
    cursor = len(q) - 1
    while cursor > 0:
        cursor = int(predecessor[cursor])
        reverse_indices.append(cursor)
    return np.asarray(list(reversed(reverse_indices)), dtype=np.int64)


@pytest.mark.parametrize("short_target", [False, True])
def test_execution_subset_matches_frozen_reference_randomized(short_target):
    rng = np.random.default_rng(20260728 + int(short_target))
    canonical_names = np.asarray(_FakeBackend().joint_names, dtype=object)
    canonical_index = {
        str(name): index for index, name in enumerate(canonical_names)
    }
    active_indices = [
        canonical_index[name] for name in WHOLE_BODY_ACTIVE_JOINT_NAMES
    ]
    locked_indices = [
        canonical_index[name] for name in WHOLE_BODY_LOCKED_JOINT_NAMES
    ]
    for _case in range(40):
        waypoint_count = int(rng.integers(2, 56))
        q = np.zeros(
            (waypoint_count, len(canonical_names)),
            dtype=np.float32,
        )
        q[:, locked_indices] = rng.uniform(
            -0.5,
            0.5,
            size=(1, len(locked_indices)),
        ).astype(np.float32)
        q[:, active_indices] = np.cumsum(
            rng.uniform(
                -0.003,
                0.003,
                size=(waypoint_count, len(active_indices)),
            ),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        yaw_index = canonical_index["base_footprint_rz_joint"]
        q[:, yaw_index] = np.asarray(
            [
                _wrap_angle(value)
                for value in q[:, yaw_index]
            ],
            dtype=np.float32,
        )
        positions = np.cumsum(
            rng.uniform(-0.0007, 0.0007, size=(waypoint_count, 3)),
            axis=0,
        )
        angles = np.cumsum(
            rng.uniform(-0.006, 0.006, size=waypoint_count)
        )
        quaternions = np.column_stack(
            [
                np.zeros(waypoint_count),
                np.zeros(waypoint_count),
                np.sin(angles * 0.5),
                np.cos(angles * 0.5),
            ]
        )
        quaternions[::3] *= -1.0
        permutation = rng.permutation(len(canonical_names))
        names = tuple(str(name) for name in canonical_names[permutation])
        q = q[:, permutation]

        expected = _reference_whole_body_execution_subset_indices(
            q,
            positions,
            quaternions,
            joint_names=names,
            short_target=short_target,
        )
        actual = _whole_body_execution_subset_indices(
            q,
            positions,
            quaternions,
            joint_names=names,
            short_target=short_target,
        )

        np.testing.assert_array_equal(actual, expected)
        assert _whole_body_execution_step_report(
            q[actual],
            positions[actual],
            quaternions[actual],
            joint_names=names,
            short_target=short_target,
        )["admitted"] is True


@pytest.mark.parametrize(
    ("gate", "short_target"),
    [
        ("base_x", True),
        ("base_y", True),
        ("base_yaw", True),
        ("articulation", True),
        ("eef", True),
        ("eef", False),
        ("orientation", True),
    ],
)
def test_execution_subset_matches_reference_at_gate_boundaries(
    gate,
    short_target,
):
    names = tuple(_FakeBackend().joint_names)
    by_name = {name: index for index, name in enumerate(names)}
    q = np.zeros((3, len(names)), dtype=np.float32)
    positions = np.zeros((3, 3), dtype=np.float64)
    quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        3,
        axis=0,
    )
    if gate == "base_x":
        q[:, by_name["base_footprint_x_joint"]] = [0.0, 0.003, 0.007502]
    elif gate == "base_y":
        q[:, by_name["base_footprint_y_joint"]] = [0.0, 0.003, 0.007502]
    elif gate == "base_yaw":
        q[:, by_name["base_footprint_rz_joint"]] = [
            0.0,
            math.radians(0.5),
            math.radians(1.0) + 2e-6,
        ]
    elif gate == "articulation":
        q[:, by_name["left_arm_joint1"]] = [0.0, 0.015, 0.030002]
    elif gate == "eef":
        limit = (
            WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
            if short_target
            else 0.0029
        )
        positions[:, 0] = [0.0, limit * 0.5, limit + 2e-6]
    else:
        limit = math.radians(1.25)
        angles = np.asarray([0.0, limit * 0.5, limit + 2e-6])
        quaternions[:, 2] = np.sin(angles * 0.5)
        quaternions[:, 3] = np.cos(angles * 0.5)

    expected = _reference_whole_body_execution_subset_indices(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=short_target,
    )
    actual = _whole_body_execution_subset_indices(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=short_target,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [0, 1, 2])


def test_execution_subset_reference_preserves_wrap_and_locked_joint_semantics():
    names = tuple(_FakeBackend().joint_names)
    by_name = {name: index for index, name in enumerate(names)}
    q = np.zeros((3, len(names)), dtype=np.float32)
    q[:, by_name["base_footprint_rz_joint"]] = [
        math.pi - 0.005,
        -math.pi,
        -math.pi + 0.005,
    ]
    q[1, by_name["left_gripper_finger_joint1"]] = np.float32(1e-4)
    positions = np.zeros((3, 3), dtype=np.float64)
    quaternions = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected = _reference_whole_body_execution_subset_indices(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )
    actual = _whole_body_execution_subset_indices(
        q,
        positions,
        quaternions,
        joint_names=names,
        short_target=True,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [0, 2])

    q[-1, by_name["left_gripper_finger_joint1"]] = np.float32(2e-7)
    with pytest.raises(RuntimeError, match="no execution subset"):
        _reference_whole_body_execution_subset_indices(
            q,
            positions,
            quaternions,
            joint_names=names,
            short_target=True,
        )
    with pytest.raises(RuntimeError, match="no execution subset"):
        _whole_body_execution_subset_indices(
            q,
            positions,
            quaternions,
            joint_names=names,
            short_target=True,
        )


def test_sparse_short_eef_admission_uses_6mm_but_dense_stays_2_2mm():
    quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        2,
        axis=0,
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [0.006, 0.0, 0.0]])

    dense = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=positions[0],
        target_xyz=positions[-1],
        target_quat_xyzw=quaternions[-1],
    )
    sparse = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=positions[0],
        target_xyz=positions[-1],
        target_quat_xyzw=quaternions[-1],
        short_cartesian_step_limit_m=(
            WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
        ),
    )
    over_limit_positions = positions.copy()
    over_limit_positions[-1, 0] += 2e-6
    sparse_over_limit = _eef_pose_path_admission_report(
        over_limit_positions,
        quaternions,
        call_start_xyz=over_limit_positions[0],
        target_xyz=over_limit_positions[-1],
        target_quat_xyzw=quaternions[-1],
        short_cartesian_step_limit_m=(
            WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
        ),
    )

    assert dense["admitted"] is False
    assert dense["short_target_cartesian_step_limit_m"] == pytest.approx(
        0.0022
    )
    assert sparse["admitted"] is True
    assert sparse["all_target_cartesian_step_limit_m"] == pytest.approx(0.006)
    assert sparse["short_target_cartesian_step_limit_m"] == pytest.approx(
        0.006
    )
    assert sparse_over_limit["admitted"] is False
    assert sparse_over_limit["checks"]["all_target_cartesian_step"] is False
    assert sparse_over_limit["checks"]["short_target_cartesian_step"] is False

    long_positions = np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]])
    long_sparse = _eef_pose_path_admission_report(
        long_positions,
        quaternions,
        call_start_xyz=long_positions[0],
        target_xyz=long_positions[-1],
        target_quat_xyzw=quaternions[-1],
        short_cartesian_step_limit_m=(
            WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
        ),
    )
    assert long_sparse["short_target"] is False
    assert long_sparse["all_target_cartesian_step_limit_m"] == pytest.approx(
        0.0029
    )
    assert long_sparse["checks"]["all_target_cartesian_step"] is False


@pytest.mark.parametrize(
    ("goal_index", "goal_delta"),
    [
        (0, 0.05),
        (5, math.radians(5.0)),
    ],
)
def test_minimum_jerk_base_jog_uses_six_commands(goal_index, goal_delta):
    start = np.zeros(28, dtype=np.float64)
    goal = start.copy()
    goal[goal_index] = goal_delta

    result, metrics = _minimum_jerk_base_execution_trajectory(
        np.stack([start, goal]),
        base_indices=list(range(6)),
    )

    assert BASE_EXECUTION_XY_STEP_M == pytest.approx(0.015)
    assert BASE_EXECUTION_YAW_STEP_RAD == pytest.approx(math.radians(1.5))
    assert BASE_EXECUTION_XY_STEP_M < TRACKING_HARD_BASE_XY_ERROR_M
    assert BASE_EXECUTION_YAW_STEP_RAD < TRACKING_HARD_BASE_YAW_ERROR_RAD
    assert len(result) - 1 == 6
    assert metrics["execution_waypoints"] == 6


def test_minimum_jerk_base_resampler_bounds_steps_and_eases_endpoints():
    base_indices = list(range(6))
    start = np.linspace(-0.4, 0.4, 28, dtype=np.float64)
    goal = start.copy()
    goal[0] += 0.05
    goal[1] -= 0.02
    goal[5] += math.radians(5.0)

    result, metrics = _minimum_jerk_base_execution_trajectory(
        np.stack([start, goal]),
        base_indices=base_indices,
    )

    np.testing.assert_allclose(result[0], start)
    np.testing.assert_allclose(result[-1], goal)
    xy_steps = np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1)
    yaw_steps = np.abs(np.diff(result[:, 5]))
    assert np.max(xy_steps) <= BASE_EXECUTION_XY_STEP_M + 1e-7
    assert np.max(yaw_steps) <= BASE_EXECUTION_YAW_STEP_RAD + 1e-7
    assert xy_steps[0] < 0.1 * np.max(xy_steps)
    assert xy_steps[-1] < 0.1 * np.max(xy_steps)
    assert yaw_steps[0] < 0.1 * np.max(yaw_steps)
    assert yaw_steps[-1] < 0.1 * np.max(yaw_steps)
    locked = [index for index in range(28) if index not in {0, 1, 5}]
    np.testing.assert_array_equal(
        result[:, locked],
        np.repeat(result[:1, locked], len(result), axis=0),
    )
    assert metrics["method"] == "quintic_minimum_jerk"
    assert metrics["profile"] == "10u^3-15u^4+6u^5"
    assert metrics["analytic_endpoint_velocity_zero"] is True
    assert metrics["analytic_endpoint_acceleration_zero"] is True
    assert metrics["measured_max_xy_step_m"] == pytest.approx(
        np.max(xy_steps), abs=1e-8
    )
    assert metrics["measured_max_yaw_step_rad"] == pytest.approx(
        np.max(yaw_steps), abs=1e-8
    )


def test_minimum_jerk_base_resampler_uses_shortest_yaw_arc():
    start = np.zeros(28, dtype=np.float64)
    start[5] = math.pi - math.radians(0.25)
    goal = start.copy()
    goal[5] = -math.pi + math.radians(0.25)

    result, metrics = _minimum_jerk_base_execution_trajectory(
        np.stack([start, goal]),
        base_indices=list(range(6)),
    )

    wrapped_steps = np.asarray(
        [
            abs((float(delta) + math.pi) % (2.0 * math.pi) - math.pi)
            for delta in np.diff(result[:, 5])
        ]
    )
    wrapped_endpoint_error = (
        float(result[-1, 5] - goal[5]) + math.pi
    ) % (2.0 * math.pi) - math.pi
    assert abs(wrapped_endpoint_error) <= 1e-6
    assert np.max(wrapped_steps) <= BASE_EXECUTION_YAW_STEP_RAD + 1e-7
    assert metrics["measured_max_yaw_step_rad"] <= (
        BASE_EXECUTION_YAW_STEP_RAD + 1e-9
    )


def test_minimum_jerk_base_resampler_rejects_locked_joint_motion():
    start = np.zeros(28, dtype=np.float64)
    goal = start.copy()
    goal[6] = 0.001

    with pytest.raises(ValueError, match="changed locked joint"):
        _minimum_jerk_base_execution_trajectory(
            np.stack([start, goal]),
            base_indices=list(range(6)),
        )


def test_minimum_jerk_base_resampler_softens_direction_reversal():
    start = np.zeros(28, dtype=np.float64)
    forward_goal = start.copy()
    forward_goal[0] = 0.05
    forward, _ = _minimum_jerk_base_execution_trajectory(
        np.stack([start, forward_goal]),
        base_indices=list(range(6)),
    )
    backward_goal = forward_goal.copy()
    backward_goal[0] = 0.0
    backward, _ = _minimum_jerk_base_execution_trajectory(
        np.stack([forward_goal, backward_goal]),
        base_indices=list(range(6)),
    )

    last_forward_delta = float(forward[-1, 0] - forward[-2, 0])
    first_backward_delta = float(backward[1, 0] - backward[0, 0])
    reversal_jump = abs(first_backward_delta - last_forward_delta)

    assert reversal_jump < 0.25 * BASE_EXECUTION_XY_STEP_M


def test_navigation_isolation_keeps_one_degree_roll_pitch_fail_closed():
    class Robot:
        base_idx = list(range(6))
        trunk_control_idx = [6, 7, 8, 9]
        arm_control_idx = {
            "left": list(range(10, 17)),
            "right": list(range(17, 24)),
        }

        def __init__(self):
            self.q = np.zeros(28, dtype=np.float64)

        def get_joint_positions(self):
            return self.q.copy()

    robot = Robot()
    backend = RealCuroboBackend(None)
    backend._robot = robot
    backend.env_facade = SimpleNamespace(
        _gripper_latch={"left": 0.0, "right": 0.0}
    )
    backend.get_attached_object = lambda _hand: None
    reference = backend.capture_navigation_isolation_reference()
    robot.q[3] = math.radians(1.0) + 1e-6

    report = backend.navigation_isolation_report(
        action=np.zeros(23, dtype=np.float32),
        reference=reference,
    )

    assert report["available"] is True
    assert report["ok"] is False
    assert report["checks"]["base_roll_pitch_locked"] is False
    assert report["thresholds"]["base_roll_pitch_rad"] == pytest.approx(
        math.radians(1.0)
    )


def test_whole_body_path_canonicalizes_two_pi_yaw_before_execution_digest():
    full_names = (
        "base_footprint_x_joint",
        "base_footprint_y_joint",
        "base_footprint_z_joint",
        "base_footprint_rx_joint",
        "base_footprint_ry_joint",
        "base_footprint_rz_joint",
        *tuple(f"torso_joint{i}" for i in range(1, 5)),
        *tuple(f"left_arm_joint{i}" for i in range(1, 8)),
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        *tuple(f"right_arm_joint{i}" for i in range(1, 8)),
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    )
    full_index = {name: index for index, name in enumerate(full_names)}
    active_names = tuple(WHOLE_BODY_ACTIVE_JOINT_NAMES)
    active_index = {name: index for index, name in enumerate(active_names)}
    start_q = np.linspace(-0.2, 0.2, len(full_names), dtype=np.float32)
    start_yaw = math.pi - 0.05
    start_q[full_index["base_footprint_rz_joint"]] = start_yaw
    values = np.stack(
        [
            np.asarray([start_q[full_index[name]] for name in active_names])
            for _ in range(3)
        ]
    )
    yaw_column = active_index["base_footprint_rz_joint"]
    raw_yaw = np.asarray(
        [
            start_yaw - 2.0 * math.pi,
            start_yaw + 0.04 + 2.0 * math.pi,
            start_yaw + 0.08 - 2.0 * math.pi,
        ],
        dtype=np.float32,
    )
    values[:, yaw_column] = raw_yaw
    values[:, active_index["right_arm_joint1"]] += np.asarray(
        [0.0, 0.01, 0.02], dtype=np.float32
    )
    path = SimpleNamespace(joint_names=list(active_names), position=values)
    generator = SimpleNamespace(
        robot_joint_names=list(full_names),
        mg={
            "default": SimpleNamespace(
                kinematics=SimpleNamespace(joint_names=list(active_names))
            )
        },
    )
    robot = SimpleNamespace(joints=dict.fromkeys(full_names))
    backend = RealCuroboBackend(None)
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default")

    merged, report = backend._whole_body_path_to_full_joint_trajectory(
        generator,
        robot,
        path,
        start_q=start_q,
    )
    merged, continuity = _canonicalize_whole_body_base_yaw_trajectory(
        merged,
        joint_names=full_names,
        call_start_q=start_q,
    )
    report["base_yaw_continuity"] = continuity

    yaw_index = full_index["base_footprint_rz_joint"]
    canonical_yaw = merged[:, yaw_index].astype(np.float64)
    np.testing.assert_allclose(
        canonical_yaw,
        start_yaw + np.asarray([0.0, 0.04, 0.08]),
        atol=1e-6,
    )
    continuity = report["base_yaw_continuity"]
    assert continuity["branch_correction_count"] == 3
    assert continuity["raw_max_step_rad"] > 2.0 * math.pi
    assert continuity["canonical_max_step_rad"] == pytest.approx(0.04, abs=1e-6)
    for raw, canonical in zip(raw_yaw, canonical_yaw, strict=True):
        assert abs((float(raw - canonical) + math.pi) % (2.0 * math.pi) - math.pi) < (
            1e-6
        )
    for name in WHOLE_BODY_LOCKED_JOINT_NAMES:
        np.testing.assert_array_equal(
            merged[:, full_index[name]],
            np.repeat(start_q[full_index[name]], len(merged)),
        )

    with_start = np.vstack([start_q, merged])
    dense = _interpolate_joint_trajectory(
        with_start,
        max_inter_dist=0.0075,
    )
    execution = _interpolate_whole_body_execution_trajectory(
        with_start,
        joint_names=full_names,
    )[1:]
    assert np.max(np.abs(np.diff(dense[:, yaw_index]))) <= 0.0075 + 1e-6
    assert np.max(np.abs(np.diff(execution[:, yaw_index]))) <= (
        math.radians(1.0) + 1e-6
    )
    assert np.sum(np.abs(np.diff(execution[:, yaw_index]))) < 0.1
    execution_digest = hashlib.sha256(
        np.ascontiguousarray(execution, dtype=np.float32).tobytes()
    ).hexdigest()
    branched_execution = execution.copy()
    branched_execution[:, yaw_index] += np.float32(2.0 * math.pi)
    assert execution_digest != hashlib.sha256(
        np.ascontiguousarray(branched_execution, dtype=np.float32).tobytes()
    ).hexdigest()


def test_whole_body_eef_path_rejects_short_target_with_large_detour():
    backend = RealCuroboBackend(None)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.03, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    backend._curobo_eef_poses = lambda _generator, _q: (
        positions,
        np.repeat([[0.0, 0.0, 0.0, 1.0]], len(positions), axis=0),
    )

    report = backend._whole_body_eef_path_report(
        object(),
        np.zeros((3, 28), dtype=np.float32),
        call_start_xyz=np.zeros(3),
        target_xyz=np.asarray([0.03, 0.0, 0.0]),
    )

    assert report["available"] is True
    assert report["admitted"] is False
    assert report["short_target"] is True
    assert report["max_start_excursion_m"] == pytest.approx(0.7)
    assert report["checks"]["short_target_start_excursion"] is False
    assert report["terminal_error_m"] == pytest.approx(0.0)


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


def _grouped_dynamics_report(
    *,
    base_translation=0.0,
    base_yaw=0.0,
    articulation=0.0,
    ok=True,
):
    values = {
        "base_translation": (
            float(base_translation),
            "m/s",
            ("base_footprint_x_joint", "base_footprint_y_joint"),
        ),
        "base_yaw": (
            float(base_yaw),
            "rad/s",
            ("base_footprint_rz_joint",),
        ),
        "articulation": (
            float(articulation),
            "rad/s",
            (
                *tuple(f"torso_joint{index}" for index in range(1, 5)),
                *tuple(f"left_arm_joint{index}" for index in range(1, 8)),
                *tuple(f"right_arm_joint{index}" for index in range(1, 8)),
            ),
        ),
    }
    indices = {
        "base_translation": [0, 1],
        "base_yaw": [5],
        "articulation": list(range(6, 24)),
    }
    groups = {}
    for name, (maximum, unit, joint_names) in values.items():
        groups[name] = {
            "available": True,
            "unit": unit,
            "dof_count": len(joint_names),
            "dof_indices": indices[name],
            "joint_names": list(joint_names),
            "max_actual_velocity": maximum,
            "max_actual_velocity_joint": joint_names[0],
            "max_velocity_limit": 1.0,
            "max_velocity_ratio": maximum,
            "within_control_limit": bool(ok),
        }
    maximum_name = max(values, key=lambda name: values[name][0])
    return {
        "available": True,
        "ok": bool(ok),
        "max_actual_velocity": values[maximum_name][0],
        "max_actual_velocity_joint": groups[maximum_name][
            "max_actual_velocity_joint"
        ],
        "max_velocity_limit": 1.0,
        "max_velocity_ratio": values[maximum_name][0],
        "velocity_groups": groups,
    }


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
        self.joint_names = (
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_z_joint",
            "base_footprint_rx_joint",
            "base_footprint_ry_joint",
            "base_footprint_rz_joint",
            *tuple(f"torso_joint{index}" for index in range(1, 5)),
            *tuple(f"left_arm_joint{index}" for index in range(1, 8)),
            "left_gripper_finger_joint1",
            "left_gripper_finger_joint2",
            *tuple(f"right_arm_joint{index}" for index in range(1, 8)),
            "right_gripper_finger_joint1",
            "right_gripper_finger_joint2",
        )
        self.whole_body_plan_calls = []
        self.whole_body_hold_calls = []
        self.whole_body_tracking_calls = []
        self.execution_eef_positions = np.empty((0, 3), dtype=np.float32)
        self.execution_eef_quaternions = np.empty((0, 4), dtype=np.float32)
        self.dense_eef_positions = np.empty((0, 3), dtype=np.float32)
        self.dense_eef_quaternions = np.empty((0, 4), dtype=np.float32)
        self.execution_eef_index = 0
        self.next_execution_eef_pose = None
        self.next_execution_eef_quat = None
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
        search_profile=WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
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
                "search_profile": str(search_profile),
                "selected_attachment": attached_obj,
                "attachments_by_hand": attachments_by_hand,
            }
        )
        nominal_max_target_error = float(np.linalg.norm(self.pose - self.target))
        waypoint_count = max(
            20,
            int(math.ceil(nominal_max_target_error / 0.0028)),
        )
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
        self.execution_eef_positions = np.ascontiguousarray(
            np.linspace(
                self.pose,
                self.target,
                waypoint_count + 1,
                dtype=np.float64,
            )[1:],
            dtype=np.float32,
        )
        self.execution_eef_index = 0
        self.next_execution_eef_pose = None
        self.next_execution_eef_quat = None
        execution_quat = self.quat if self.target_quat is None else self.target_quat
        start_quat = np.asarray(self.quat, dtype=np.float64)
        end_quat = np.asarray(execution_quat, dtype=np.float64)
        if float(np.dot(start_quat, end_quat)) < 0.0:
            end_quat = -end_quat
        interpolated_quats = []
        for fraction in np.linspace(
            1.0 / waypoint_count,
            1.0,
            waypoint_count,
        ):
            quat = (1.0 - fraction) * start_quat + fraction * end_quat
            interpolated_quats.append(quat / np.linalg.norm(quat))
        self.execution_eef_quaternions = np.ascontiguousarray(
            interpolated_quats,
            dtype=np.float32,
        )
        execution_with_start = np.vstack(
            [self.joint_positions.reshape(1, -1), trajectory]
        ).astype(np.float32)
        dense_collision_trajectory = _interpolate_joint_trajectory(
            execution_with_start,
            max_inter_dist=WHOLE_BODY_DENSE_COLLISION_STEP,
        )
        collision_dense_indices = []
        search_start = 0
        for anchor in execution_with_start:
            matches = np.flatnonzero(
                np.all(
                    dense_collision_trajectory[search_start:] == anchor,
                    axis=1,
                )
            )
            assert len(matches)
            match = search_start + int(matches[0])
            collision_dense_indices.append(match)
            search_start = match + 1
        collision_dense_indices = np.asarray(
            collision_dense_indices,
            dtype=np.int64,
        )
        # The source collision path intentionally has additional ordered rows
        # so the default fake exercises a genuine execution subset instead of
        # the old dense == [start; execution] shortcut.
        source_dense_rows = [execution_with_start[0]]
        source_dense_indices = [0]
        for row in execution_with_start[1:]:
            source_dense_rows.extend([row.copy(), row.copy()])
            source_dense_indices.append(len(source_dense_rows) - 1)
        source_dense_trajectory = np.asarray(
            source_dense_rows,
            dtype=np.float32,
        )
        source_dense_indices = np.asarray(source_dense_indices, dtype=np.int64)
        eef_position_anchors = np.vstack(
            [
                self.pose.reshape(1, 3),
                self.execution_eef_positions,
            ]
        )
        eef_quaternion_anchors = np.vstack(
            [
                start_quat.reshape(1, 4),
                self.execution_eef_quaternions,
            ]
        )
        dense_eef_positions = []
        dense_eef_quaternions = []
        for q_start, q_end, xyz_start, xyz_end, quat_start, quat_end in zip(
            execution_with_start[:-1],
            execution_with_start[1:],
            eef_position_anchors[:-1],
            eef_position_anchors[1:],
            eef_quaternion_anchors[:-1],
            eef_quaternion_anchors[1:],
            strict=True,
        ):
            intervals = max(
                1,
                int(
                    math.ceil(
                        float(np.max(np.abs(q_end - q_start)))
                        / WHOLE_BODY_DENSE_COLLISION_STEP
                    )
                ),
            )
            for interval_index in range(intervals):
                alpha = interval_index / intervals
                dense_eef_positions.append(
                    xyz_start + alpha * (xyz_end - xyz_start)
                )
                quat = quat_start + alpha * (quat_end - quat_start)
                dense_eef_quaternions.append(quat / np.linalg.norm(quat))
        dense_eef_positions.append(eef_position_anchors[-1])
        dense_eef_quaternions.append(eef_quaternion_anchors[-1])
        self.dense_eef_positions = np.ascontiguousarray(
            dense_eef_positions,
            dtype=np.float32,
        )
        self.dense_eef_quaternions = np.ascontiguousarray(
            dense_eef_quaternions,
            dtype=np.float32,
        )
        assert len(self.dense_eef_positions) == len(dense_collision_trajectory)

        def digest_float32(values):
            return hashlib.sha256(
                np.ascontiguousarray(values, dtype=np.float32).tobytes()
            ).hexdigest()

        def digest_indices(values):
            return hashlib.sha256(
                np.ascontiguousarray(values, dtype="<i8").tobytes()
            ).hexdigest()

        certificate = {
            "schema_version": 3,
            "collision_lineage_mode": (
                "source_dense_execution_subset_recertified_v1"
            ),
            "q_encoding": "float32-c-order",
            "trajectory_sha256": digest_float32(trajectory),
            "execution_trajectory_sha256": digest_float32(trajectory),
            "start_q_sha256": hashlib.sha256(
                np.ascontiguousarray(self.joint_positions, dtype=np.float32).tobytes()
            ).hexdigest(),
            "waypoint_count": waypoint_count,
            "execution_waypoint_count": waypoint_count,
            "execution_includes_start": False,
            "q_dimension": q_dimension,
            "joint_name_layout_sha256": hashlib.sha256(
                json.dumps(
                    list(self.joint_names[:q_dimension]),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "active_dof_count": 21,
            "selected_eef_goal_count": 1,
            "inactive_eef_goal_count": 0,
            "attachment_hand_count": 2,
            "world_collision_check": True,
            "self_collision_check": True,
            "post_interpolation_check": True,
            "collision_free_waypoints": len(dense_collision_trajectory),
            "dense_collision_waypoint_count": len(dense_collision_trajectory),
            "dense_collision_checked_waypoint_count": len(
                dense_collision_trajectory
            ),
            "dense_collision_includes_start": True,
            "dense_collision_trajectory_sha256": digest_float32(
                dense_collision_trajectory
            ),
            "dense_collision_trajectory": dense_collision_trajectory.tolist(),
            "collision_densification_method": "joint_linear_max_inter_dist",
            "collision_densification_joint_step_rad": (
                WHOLE_BODY_DENSE_COLLISION_STEP
            ),
            "source_dense_waypoint_count": len(source_dense_trajectory),
            "source_dense_includes_start": True,
            "source_dense_trajectory_sha256": digest_float32(
                source_dense_trajectory
            ),
            "source_dense_trajectory": source_dense_trajectory.tolist(),
            "execution_source_dense_indices": source_dense_indices.tolist(),
            "execution_source_dense_indices_sha256": digest_indices(
                source_dense_indices
            ),
            "execution_source_terminal_index": int(source_dense_indices[-1]),
            "execution_collision_dense_indices": (
                collision_dense_indices.tolist()
            ),
            "execution_collision_dense_indices_sha256": digest_indices(
                collision_dense_indices
            ),
            "execution_collision_terminal_index": int(
                collision_dense_indices[-1]
            ),
            "selected_hand": hand,
            "selected_target_xyz_sha256": hashlib.sha256(
                np.ascontiguousarray(self.target, dtype=np.float32).tobytes()
            ).hexdigest(),
            "selected_target_quat_xyzw_sha256": (
                None
                if target_quat_xyzw is None
                else hashlib.sha256(
                    np.ascontiguousarray(
                        execution_quat, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
            ),
            "selected_eef_path_admitted": True,
            "selected_eef_short_target": (
                nominal_max_target_error <= 0.036 + 1e-9
            ),
            "selected_eef_positions_sha256": digest_float32(
                self.dense_eef_positions
            ),
            "selected_eef_dense_positions_sha256": digest_float32(
                self.dense_eef_positions
            ),
            "selected_eef_dense_quaternions_sha256": digest_float32(
                self.dense_eef_quaternions
            ),
            "selected_eef_dense_positions": self.dense_eef_positions.tolist(),
            "selected_eef_dense_quaternions_xyzw": (
                self.dense_eef_quaternions.tolist()
            ),
            "selected_eef_nominal_max_target_error_m": (
                nominal_max_target_error
            ),
            "selected_eef_execution_positions_sha256": digest_float32(
                self.execution_eef_positions
            ),
            "selected_eef_execution_quaternions_sha256": digest_float32(
                self.execution_eef_quaternions
            ),
            "selected_eef_execution_positions": (
                self.execution_eef_positions.tolist()
            ),
            "selected_eef_execution_quaternions_xyzw": (
                self.execution_eef_quaternions.tolist()
            ),
            "selected_eef_live_waypoint_position_tolerance_m": 0.005,
            "selected_eef_live_waypoint_orientation_tolerance_rad": (
                math.radians(1.0)
            ),
            "selected_eef_controller_response_margin_m": (
                WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
            ),
            "selected_eef_numerical_margin_m": (
                WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
            ),
            "selected_eef_prospective_guard_margin_m": (
                WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
            ),
            "execution_base_xy_step_limit_m": (
                WHOLE_BODY_EXECUTION_BASE_XY_STEP_M
            ),
            "execution_base_yaw_step_limit_rad": (
                WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD
            ),
            "execution_articulation_step_limit_rad": (
                WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
            ),
            "terminal_command_limit": TERMINAL_COMMAND_LIMIT,
            "terminal_eef_position_tolerance_m": (
                EEF_TERMINAL_POSITION_TOLERANCE_M
            ),
            "terminal_eef_orientation_tolerance_rad": (
                EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD
            ),
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
        result = {
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
        return self._refresh_whole_body_certificate(result)

    def _refresh_whole_body_certificate(self, result):
        trajectory = np.ascontiguousarray(
            result["joint_trajectory"],
            dtype=np.float32,
        )
        execution_with_start = np.vstack(
            [self.joint_positions.reshape(1, -1), trajectory]
        ).astype(np.float32)
        dense = _interpolate_joint_trajectory(
            execution_with_start,
            max_inter_dist=WHOLE_BODY_DENSE_COLLISION_STEP,
        )

        def ordered_indices(rows, anchors):
            indices = [0]
            cursor = 1
            for anchor_index, anchor in enumerate(anchors[1:], start=1):
                if anchor_index == len(anchors) - 1:
                    match = len(rows) - 1
                else:
                    matches = np.flatnonzero(
                        np.all(rows[cursor:-1] == anchor, axis=1)
                    )
                    assert len(matches)
                    match = cursor + int(matches[0])
                assert np.array_equal(rows[match], anchor)
                indices.append(match)
                cursor = match + 1
            return np.asarray(indices, dtype=np.int64)

        collision_indices = ordered_indices(dense, execution_with_start)
        source_rows = [execution_with_start[0]]
        source_indices = [0]
        for row in execution_with_start[1:]:
            source_rows.extend([row.copy(), row.copy()])
            source_indices.append(len(source_rows) - 1)
        source_dense = np.asarray(source_rows, dtype=np.float32)
        source_indices = np.asarray(source_indices, dtype=np.int64)

        execution_position_anchors = np.vstack(
            [self.pose.reshape(1, 3), self.execution_eef_positions]
        )
        execution_quaternion_anchors = np.vstack(
            [self.quat.reshape(1, 4), self.execution_eef_quaternions]
        )
        dense_positions = []
        dense_quaternions = []
        for q_start, q_end, xyz_start, xyz_end, quat_start, quat_end in zip(
            execution_with_start[:-1],
            execution_with_start[1:],
            execution_position_anchors[:-1],
            execution_position_anchors[1:],
            execution_quaternion_anchors[:-1],
            execution_quaternion_anchors[1:],
            strict=True,
        ):
            intervals = max(
                1,
                int(
                    math.ceil(
                        float(np.max(np.abs(q_end - q_start)))
                        / WHOLE_BODY_DENSE_COLLISION_STEP
                    )
                ),
            )
            for interval_index in range(intervals):
                alpha = interval_index / intervals
                dense_positions.append(
                    xyz_start + alpha * (xyz_end - xyz_start)
                )
                quat = quat_start + alpha * (quat_end - quat_start)
                dense_quaternions.append(quat / np.linalg.norm(quat))
        dense_positions.append(execution_position_anchors[-1])
        dense_quaternions.append(execution_quaternion_anchors[-1])
        self.dense_eef_positions = np.ascontiguousarray(
            dense_positions,
            dtype=np.float32,
        )
        self.dense_eef_quaternions = np.ascontiguousarray(
            dense_quaternions,
            dtype=np.float32,
        )
        assert len(self.dense_eef_positions) == len(dense)

        def digest_float32(values):
            return hashlib.sha256(
                np.ascontiguousarray(values, dtype=np.float32).tobytes()
            ).hexdigest()

        def digest_indices(values):
            return hashlib.sha256(
                np.ascontiguousarray(values, dtype="<i8").tobytes()
            ).hexdigest()

        certificate = result["whole_body_certificate"]
        certificate.update(
            {
                "trajectory_sha256": digest_float32(trajectory),
                "execution_trajectory_sha256": digest_float32(trajectory),
                "waypoint_count": len(trajectory),
                "execution_waypoint_count": len(trajectory),
                "collision_free_waypoints": len(dense),
                "dense_collision_waypoint_count": len(dense),
                "dense_collision_checked_waypoint_count": len(dense),
                "dense_collision_trajectory_sha256": digest_float32(dense),
                "dense_collision_trajectory": dense.tolist(),
                "source_dense_waypoint_count": len(source_dense),
                "source_dense_trajectory_sha256": digest_float32(source_dense),
                "source_dense_trajectory": source_dense.tolist(),
                "execution_source_dense_indices": source_indices.tolist(),
                "execution_source_dense_indices_sha256": digest_indices(
                    source_indices
                ),
                "execution_source_terminal_index": int(source_indices[-1]),
                "execution_collision_dense_indices": (
                    collision_indices.tolist()
                ),
                "execution_collision_dense_indices_sha256": digest_indices(
                    collision_indices
                ),
                "execution_collision_terminal_index": int(
                    collision_indices[-1]
                ),
                "selected_eef_positions_sha256": digest_float32(
                    self.dense_eef_positions
                ),
                "selected_eef_dense_positions_sha256": digest_float32(
                    self.dense_eef_positions
                ),
                "selected_eef_dense_quaternions_sha256": digest_float32(
                    self.dense_eef_quaternions
                ),
                "selected_eef_dense_positions": (
                    self.dense_eef_positions.tolist()
                ),
                "selected_eef_dense_quaternions_xyzw": (
                    self.dense_eef_quaternions.tolist()
                ),
                "selected_eef_execution_positions_sha256": digest_float32(
                    self.execution_eef_positions
                ),
                "selected_eef_execution_quaternions_sha256": digest_float32(
                    self.execution_eef_quaternions
                ),
                "selected_eef_execution_positions": (
                    self.execution_eef_positions.tolist()
                ),
                "selected_eef_execution_quaternions_xyzw": (
                    self.execution_eef_quaternions.tolist()
                ),
            }
        )
        result["metrics"]["whole_body_certificate"] = certificate
        return result

    def get_eef_pose(self, hand):
        return self.pose.copy(), self.quat.copy()

    def whole_body_eef_poses(self, *, hand, q_trajectory):
        del hand
        waypoint_count = np.asarray(q_trajectory).shape[0]
        if waypoint_count == len(self.execution_eef_positions):
            return (
                self.execution_eef_positions.copy(),
                self.execution_eef_quaternions.copy(),
            )
        if waypoint_count == len(self.execution_eef_positions) + 1:
            return (
                self.dense_eef_positions.copy(),
                self.dense_eef_quaternions.copy(),
            )
        if waypoint_count == len(self.dense_eef_positions):
            return (
                self.dense_eef_positions.copy(),
                self.dense_eef_quaternions.copy(),
            )
        raise AssertionError(
            f"unexpected whole-body FK path length {waypoint_count}"
        )

    def get_joint_positions(self):
        return self.joint_positions.copy()

    def whole_body_joint_names(self, hand):
        del hand
        return self.joint_names

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
        if self.progress and self.next_execution_eef_pose is not None:
            self.pose = self.next_execution_eef_pose.copy()
            self.quat = self.next_execution_eef_quat.copy()
        elif self.progress and self.target is not None:
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
        self,
        *,
        baseline,
        expected_attachments_by_hand,
        allowed_expected_contact=None,
    ):
        del baseline, expected_attachments_by_hand, allowed_expected_contact
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

    @staticmethod
    def dynamics_report():
        return _grouped_dynamics_report()

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
            index = min(
                self.execution_eef_index,
                len(self.execution_eef_positions) - 1,
            )
            self.next_execution_eef_pose = self.execution_eef_positions[
                index
            ].astype(np.float64)
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                index
            ].astype(np.float64)
            self.execution_eef_index += 1
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


def _move_to_without_replan_checkpoint(
    executor,
    *,
    hand,
    target_xyz,
    **kwargs,
):
    parameters = {
        "frame": "world",
        "target_quat_xyzw": None,
        "plan_only": False,
        "position_tolerance_m": 0.02,
        "orientation_tolerance_rad": math.radians(5.0),
        "timeout_s": 240.0,
        "allow_replan_checkpoint": False,
    }
    parameters.update(kwargs)
    return executor._move_to_whole_body_impl(
        hand=hand,
        target_xyz=target_xyz,
        **parameters,
    )


def _scripted_whole_body_replan_checkpoint(
    execution_kwargs,
    *,
    final_position_error_m,
    call_start_position_error_m=None,
    checkpoint_overrides=None,
):
    trajectory = np.ascontiguousarray(
        execution_kwargs["joint_trajectory"], dtype=np.float32
    )
    assert trajectory.ndim == 2 and len(trajectory) >= 2
    certificate = execution_kwargs["whole_body_certificate"]
    call_start_error = (
        float(final_position_error_m) + 0.003
        if call_start_position_error_m is None
        else float(call_start_position_error_m)
    )
    checkpoint = {
        "commanded_index": 0,
        "post_step_index": 1,
        "q_sha256": hashlib.sha256(trajectory[0].tobytes()).hexdigest(),
        "trajectory_sha256": certificate["trajectory_sha256"],
        "call_start_position_error_m": call_start_error,
        "final_position_error_m": float(final_position_error_m),
        "position_improvement_m": (
            call_start_error - float(final_position_error_m)
        ),
        "minimum_position_improvement_m": 0.002,
        "trigger_position_improvement_m": (
            WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
        ),
        "eef_pose_settled": True,
        "dynamics_stationary": True,
        "joint_waypoint_reached": True,
    }
    checkpoint.update(checkpoint_overrides or {})
    return {
        "primitive_success": False,
        "task_success": False,
        "stop_reason": "whole_body_replan_checkpoint",
        "recoverable": True,
        "suggested_next_tool": None,
        "metrics": {
            "final_position_error_m": float(final_position_error_m),
            "trajectory_complete": False,
            "env_actions_sent": 1,
            "post_stop_env_actions": 0,
            "whole_body_replan_checkpoint": checkpoint,
        },
        "diagnostics": {"trace": []},
    }


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
    assert result["stop_reason"] == "eef_path_divergence"
    assert result["recoverable"] is False
    assert len(env.calls) == 18
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_move_to_noncheckpoint_execution_failure_never_replans():
    backend = _FakeBackend(progress=False)
    executor, env = _executor(backend)
    execute_calls = 0

    def execute(*_args, **_kwargs):
        nonlocal execute_calls
        execute_calls += 1
        return {
            "primitive_success": False,
            "task_success": False,
            "stop_reason": "stalled_tracking",
            "recoverable": True,
            "suggested_next_tool": None,
            "metrics": {
                "final_position_error_m": 0.01,
                "trajectory_complete": False,
                "env_actions_sent": 1,
            },
            "diagnostics": {"trace": []},
        }

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "stalled_tracking"
    assert execute_calls == 1
    assert len(backend.whole_body_plan_calls) == 1
    assert env.calls == []


def test_move_to_checkpoint_replans_until_eventual_success():
    backend = _FakeBackend(progress=False)
    executor, _env = _executor(backend)
    scripted_position_errors = iter([0.020, 0.018, 0.016])
    execute_calls = 0

    def execute(*_args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 4:
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "reached",
                "recoverable": True,
                "suggested_next_tool": None,
                "metrics": {
                    "final_position_error_m": 0.0,
                    "trajectory_complete": True,
                    "env_actions_sent": 1,
                },
                "diagnostics": {"trace": []},
            }
        return _scripted_whole_body_replan_checkpoint(
            kwargs,
            final_position_error_m=next(scripted_position_errors),
        )

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert execute_calls == 4
    assert len(backend.whole_body_plan_calls) == 4
    assert len(result["metrics"]["replan_rounds"]) == 4
    assert result["metrics"]["replan_rounds"][0][
        "eligible_replan_failure"
    ]["first_checkpoint"] is True
    assert result["metrics"]["replan_rounds"][1]["replan_progress"][
        "position_improved_at_least_2mm"
    ] is True
    assert result["metrics"]["replan_rounds"][2]["replan_progress"][
        "position_improved_at_least_2mm"
    ] is True
    assert result["metrics"]["replan_rounds"][2]["replan_progress"][
        "fresh_plan_tracking_comparison_used"
    ] is False


def test_move_to_checkpoint_stops_when_later_world_error_progress_is_below_2mm():
    backend = _FakeBackend(progress=False)
    executor, env = _executor(backend)
    scripted_position_errors = iter([0.020, 0.018, 0.016002])
    execute_calls = 0

    def execute(*_args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        position_error = next(scripted_position_errors)
        return _scripted_whole_body_replan_checkpoint(
            kwargs,
            final_position_error_m=position_error,
        )

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "replan_no_progress"
    assert execute_calls == 3
    assert len(backend.whole_body_plan_calls) == 3
    assert env.calls == []
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert result["metrics"]["replan_rounds"][-1]["replan_progress"][
        "position_improved_at_least_2mm"
    ] is False


@pytest.mark.parametrize(
    "checkpoint_overrides",
    [
        {"commanded_index": 999, "post_step_index": 1000},
        {"q_sha256": "0" * 64},
        {"position_improvement_m": float("nan")},
        {"post_step_index": 2},
    ],
)
def test_move_to_rejects_forged_replan_checkpoint_without_second_plan(
    checkpoint_overrides,
):
    backend = _FakeBackend(progress=False)
    executor, env = _executor(backend)
    execute_calls = 0

    def execute(*_args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        return _scripted_whole_body_replan_checkpoint(
            kwargs,
            final_position_error_m=0.02,
            checkpoint_overrides=checkpoint_overrides,
        )

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "whole_body_replan_checkpoint_invalid"
    assert execute_calls == 1
    assert len(backend.whole_body_plan_calls) == 1
    assert env.calls == []
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_move_to_replan_checkpoint_requires_zero_post_stop_actions():
    backend = _FakeBackend(progress=False)
    executor, env = _executor(backend)

    def execute(*_args, **kwargs):
        result = _scripted_whole_body_replan_checkpoint(
            kwargs,
            final_position_error_m=0.02,
        )
        result["metrics"]["post_stop_env_actions"] = 1
        return result

    executor._execute_actions = execute
    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "whole_body_replan_checkpoint_invalid"
    assert len(backend.whole_body_plan_calls) == 1
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
    assert len(backend.whole_body_plan_calls) == 1
    assert all(
        round_report.get("execution_stop_reason")
        != "whole_body_replan_checkpoint"
        for round_report in result["metrics"]["replan_rounds"]
    )
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
        for name in ("collision_report", "joint_margin_report")
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


def test_move_to_rejects_execution_step_over_limit_before_first_action():
    executor, env = _executor(_FakeBackend(bad_velocity=True))

    result = executor.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "execution_step_admitted" in result["diagnostics"]["error"]
    assert env.calls == []


@pytest.mark.parametrize(
    ("reporter_name", "outcome"),
    _REMOVED_RUNTIME_REPORTER_CASES[:-1],
    ids=(
        "collision_true",
        "collision_unavailable",
        "joint_margin_false",
        "joint_margin_unavailable",
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
        time.sleep(0.08)
        return chunk_step(actions)

    env.chunk_step = slow_chunk_step
    started = time.monotonic()
    result = executor.move_to(
        hand="left",
        target_xyz=[0.0, 0.0, 0.0],
        timeout_s=0.05,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "execution_budget_exhausted"
    assert result["metrics"]["env_actions_sent"] == 1
    assert result["metrics"]["env_actions_confirmed"] == 0
    assert result["metrics"]["env_actions_sent_exact"] is False
    assert result["metrics"]["partial_motion"] is True
    assert time.monotonic() - started < 0.4


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


@pytest.mark.parametrize("kind", ["base", "whole_body"])
def test_fast_obstacle_refresh_is_installed_on_collision_generators(
    tmp_path,
    kind,
):
    class Generator:
        def __init__(self, *_args, **_kwargs):
            pass

    robot = SimpleNamespace(
        curobo_path={"default": "default.yaml"},
        base_joint_names=[],
    )
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._lazy_imports = lambda: None
    backend._curobo_cls = Generator
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default", BASE="base")
    backend._find_robot = lambda: robot
    backend._base_config_path = lambda: tmp_path / "base.yaml"
    backend._whole_body_config_path = lambda _hand: tmp_path / "whole.yaml"
    backend._base_prismatic_workspace_limit = lambda _robot: 10.0
    installed = []
    backend._install_fast_obstacle_refresh = installed.append

    generator = backend._generator(kind=kind, hand="left")

    assert installed == [generator]


def test_fast_obstacle_refresh_full_pose_topology_ignore_and_error_fallback(
    tmp_path,
):
    class Checker:
        def __init__(self):
            self.tensor_args = object()
            self._env_mesh_names = [["/mesh/a", "/mesh/b"]]
            self._env_n_mesh = [2]
            self.pose_updates = []
            self.fail_name = None
            self.clear_calls = 0

        def update_obstacle_pose(self, **kwargs):
            self.pose_updates.append(dict(kwargs))
            if kwargs["name"] == self.fail_name:
                self.fail_name = None
                raise RuntimeError("injected pose update failure")

    class GraphPlanner:
        def __init__(self):
            self.reset_calls = 0

        def reset_buffer(self):
            self.reset_calls += 1

    class MotionGenerator:
        def __init__(self, checker):
            self.world_coll_checker = checker
            self.graph_planner = GraphPlanner()

        def clear_world_cache(self):
            self.world_coll_checker.clear_calls += 1

    class Generator:
        def __init__(self):
            self.checker = Checker()
            self.mg = {
                "default": MotionGenerator(self.checker),
                "base": MotionGenerator(self.checker),
            }
            self.full_updates = 0
            self.collision_calls = 0
            self.ignored = []

        def update_obstacles(self, ignore_objects=None):
            self.full_updates += 1
            self.ignored.append(ignore_objects)

        def check_collisions(self):
            self.collision_calls += 1
            self.update_obstacles()
            return np.asarray([False])

    snapshot_state = {
        "topology": ("topology-a",),
        "poses": (
            ("/mesh/a", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            ("/mesh/b", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    }

    def snapshot_provider(_generator, _ignore_objects):
        return {
            "topology": snapshot_state["topology"],
            "poses": snapshot_state["poses"],
            "mesh_count": len(snapshot_state["poses"]),
        }

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._install_fast_obstacle_refresh(
        generator,
        snapshot_provider=snapshot_provider,
        world_collision_type=Checker,
        pose_factory=lambda pose, _tensor_args: tuple(pose),
    )

    generator.update_obstacles()
    first = backend._obstacle_refresh_metrics(generator)
    assert generator.full_updates == 1
    assert first["mode"] == "full"
    assert first["reason"] == "first_refresh"
    assert first["count"] == 1
    assert first["fallback"] is False

    snapshot_state["poses"] = (
        ("/mesh/a", [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("/mesh/b", [1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert generator.check_collisions().tolist() == [False]
    second = backend._obstacle_refresh_metrics(generator)
    assert generator.collision_calls == 1
    assert generator.full_updates == 1
    assert [item["name"] for item in generator.checker.pose_updates] == [
        "/mesh/a",
        "/mesh/b",
    ]
    assert all(
        item["update_cpu_reference"] is True
        for item in generator.checker.pose_updates
    )
    assert second["mode"] == "pose_only"
    assert second["count"] == 2
    assert second["topology_verified"] is True
    assert second["collision_checks_skipped"] is False
    assert second["stale_pose_ttl_s"] is None
    assert second["graph_buffer_reset_count"] == 2

    snapshot_state["topology"] = ("topology-b",)
    generator.update_obstacles()
    topology = backend._obstacle_refresh_metrics(generator)
    assert generator.full_updates == 2
    assert generator.checker.clear_calls == 1
    assert topology["mode"] == "full"
    assert topology["reason"] == "topology_change"
    assert topology["count"] == 3

    generator.update_obstacles(ignore_objects=[object()])
    ignored = backend._obstacle_refresh_metrics(generator)
    assert generator.full_updates == 3
    assert generator.ignored[-1] is not None
    assert ignored["reason"] == "ignore_objects"
    assert ignored["count"] == 4

    generator.checker.fail_name = "/mesh/b"
    generator.update_obstacles()
    fallback = backend._obstacle_refresh_metrics(generator)
    assert generator.full_updates == 4
    assert generator.checker.clear_calls == 2
    assert fallback["mode"] == "full"
    assert fallback["reason"] == "fallback_full"
    assert fallback["fallback"] is True
    assert "injected pose update failure" in fallback["fallback_reason"]
    assert fallback["count"] == 5
    assert all(
        motion_gen.graph_planner.reset_calls >= 5
        for motion_gen in generator.mg.values()
    )

    backend._generators["base:left"] = generator
    backend.on_runtime_state_changed()
    generator.update_obstacles()
    retained = backend._obstacle_refresh_metrics(generator)
    assert generator.full_updates == 4
    assert retained["reason"] == "unchanged_rigid_mesh_topology"
    assert retained["count"] == 6


@_REQUIRES_TORCH
def test_fast_obstacle_refresh_batches_gpu_and_cpu_pose_updates(tmp_path):
    class Checker:
        def __init__(self):
            self.tensor_args = SimpleNamespace(device="cpu")
            self._env_mesh_names = [["/mesh/a", "/mesh/b"]]
            self._env_n_mesh = [2]
            self.world_model = SimpleNamespace(
                objects=[
                    SimpleNamespace(name="/mesh/a", pose=None),
                    SimpleNamespace(name="/mesh/b", pose=None),
                ]
            )
            self.batch_updates = []

        def update_mesh_pose(self, **kwargs):
            self.batch_updates.append(dict(kwargs))

        def update_obstacle_pose(self, **_kwargs):
            raise AssertionError("batch-capable checker used the scalar pose API")

    class MotionGenerator:
        def __init__(self, checker):
            self.world_coll_checker = checker
            self.graph_planner = SimpleNamespace(reset_buffer=lambda: None)

        def clear_world_cache(self):
            pass

    class Generator:
        def __init__(self):
            self.checker = Checker()
            self.mg = {"default": MotionGenerator(self.checker)}
            self.full_updates = 0

        def update_obstacles(self, ignore_objects=None):
            assert ignore_objects is None
            self.full_updates += 1

    snapshot = {
        "topology": ("same-topology",),
        "poses": (
            ("/mesh/a", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            ("/mesh/b", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
        "mesh_count": 2,
    }
    points_a, faces_a, points_b, faces_b = (object() for _ in range(4))
    snapshot["storage_refs"] = (
        ("/mesh/a", points_a, faces_a),
        ("/mesh/b", points_b, faces_b),
    )
    kinematic_cache_receipts = []

    def snapshot_provider(
        _generator,
        _ignored,
        *,
        kinematic_cache=None,
    ):
        kinematic_cache_receipts.append(kinematic_cache)
        snapshot["kinematic_cache"] = {
            "generation": len(kinematic_cache_receipts)
        }
        return snapshot

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._install_fast_obstacle_refresh(
        generator,
        snapshot_provider=snapshot_provider,
        world_collision_type=Checker,
        pose_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("scalar pose factory used")
        ),
        pose_batch_factory=lambda rows, _tensor_args: ("batch", rows),
    )

    generator.update_obstacles()
    snapshot["poses"] = (
        ("/mesh/a", [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("/mesh/b", [1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    generator.update_obstacles()

    assert generator.full_updates == 1
    assert len(generator.checker.batch_updates) == 1
    update = generator.checker.batch_updates[0]
    assert update["w_obj_pose"][0] == "batch"
    assert update["env_obj_idx"].tolist() == [0, 1]
    assert [
        obstacle.pose for obstacle in generator.checker.world_model.objects
    ] == [list(pose) for _name, pose in snapshot["poses"]]
    metrics = backend._obstacle_refresh_metrics(generator)
    assert metrics["mode"] == "pose_only"
    assert metrics["pose_update_mode"] == "batch_tensor_and_linear_cpu"
    assert kinematic_cache_receipts == [
        None,
        None,
        {"generation": 2},
    ]

    # Replacement is detected by object identity while the baseline retains
    # strong references, so CPython id reuse cannot create an ABA match.
    snapshot["storage_refs"] = (
        ("/mesh/a", object(), faces_a),
        ("/mesh/b", points_b, faces_b),
    )
    generator.update_obstacles()
    assert generator.full_updates == 2
    assert len(generator.checker.batch_updates) == 1
    metrics = backend._obstacle_refresh_metrics(generator)
    assert metrics["mode"] == "full"
    assert metrics["reason"] == "mesh_storage_replaced"
    assert kinematic_cache_receipts[-1] == {"generation": 3}


@_REQUIRES_TORCH
def test_mesh_topology_key_is_identical_for_full_and_pose_only_snapshots():
    import torch

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    full_key = RealCuroboBackend._mesh_topology_component(
        vertices,
        full_digest=True,
        exact_values=True,
    )
    pose_only_key = RealCuroboBackend._mesh_topology_component(
        vertices,
        full_digest=True,
        exact_values=True,
    )
    assert full_key == pose_only_key
    assert RealCuroboBackend._mesh_topology_component(
        vertices.clone(),
        full_digest=True,
        exact_values=True,
    ) == pose_only_key
    assert RealCuroboBackend._mesh_topology_component(
        torch.ones_like(vertices),
        full_digest=True,
        exact_values=True,
    ) != pose_only_key

    vertices.add_(1.0)
    mutated_key = RealCuroboBackend._mesh_topology_component(
        vertices,
        full_digest=True,
        exact_values=True,
    )
    assert mutated_key != pose_only_key


@_REQUIRES_TORCH
def test_cached_torch_mesh_topology_key_uses_identity_and_mutation_version():
    import torch

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    initial = RealCuroboBackend._mesh_topology_component(
        vertices,
        full_digest=True,
        exact_values=False,
    )
    assert (
        RealCuroboBackend._mesh_topology_component(
            vertices,
            full_digest=False,
            exact_values=False,
        )
        == initial
    )
    assert (
        RealCuroboBackend._mesh_topology_component(
            vertices.clone(),
            full_digest=False,
            exact_values=False,
        )
        != initial
    )

    vertices.add_(1.0)
    assert (
        RealCuroboBackend._mesh_topology_component(
            vertices,
            full_digest=False,
            exact_values=False,
        )
        != initial
    )


@_REQUIRES_TORCH
def test_collision_snapshot_reads_each_parent_link_once_and_reuses_local_pose(
    monkeypatch,
):
    import torch

    def pose2mat(pose):
        position, quaternion = pose
        position = torch.as_tensor(position, dtype=torch.float32)
        x, y, z, w = torch.as_tensor(quaternion, dtype=torch.float32)
        rotation = torch.tensor(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=torch.float32,
        )
        result = torch.eye(4, dtype=torch.float32)
        result[:3, :3] = rotation
        result[:3, 3] = position
        return result

    def mat2pose(matrix):
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
        yaw = torch.atan2(matrix[1, 0], matrix[0, 0])
        quaternion = torch.stack(
            [
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.sin(yaw / 2),
                torch.cos(yaw / 2),
            ]
        )
        return matrix[:3, 3].clone(), quaternion

    transform_module = ModuleType("omnigibson.utils.transform_utils")
    transform_module.pose2mat = pose2mat
    transform_module.pose_inv = torch.linalg.inv
    transform_module.mat2pose = mat2pose
    utils_module = ModuleType("omnigibson.utils")
    utils_module.__path__ = []
    utils_module.transform_utils = transform_module
    lazy_module = ModuleType("omnigibson.lazy")
    og_module = ModuleType("omnigibson")
    og_module.__path__ = []
    og_module.sim = SimpleNamespace(floor_plane=None)
    og_module.lazy = lazy_module
    monkeypatch.setitem(sys.modules, "omnigibson", og_module)
    monkeypatch.setitem(sys.modules, "omnigibson.lazy", lazy_module)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.transform_utils",
        transform_module,
    )

    class Link:
        def __init__(self, position, yaw):
            self.position = torch.tensor(position, dtype=torch.float32)
            self.yaw = float(yaw)
            self.pose_reads = 0
            self.collision_meshes = {}

        def get_position_orientation(self):
            self.pose_reads += 1
            return (
                self.position,
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        math.sin(self.yaw / 2),
                        math.cos(self.yaw / 2),
                    ],
                    dtype=torch.float32,
                ),
            )

    class Mesh:
        geom_type = "Mesh"

        def __init__(self, name, link, local_position):
            self.prim_path = name
            self.link = link
            self.local = pose2mat(
                (
                    torch.tensor(local_position, dtype=torch.float32),
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                )
            )
            self.points = torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=torch.float32,
            )
            self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
            self.pose_reads = 0

        def get_position_orientation(self):
            self.pose_reads += 1
            return mat2pose(pose2mat(self.link.get_position_orientation()) @ self.local)

        @staticmethod
        def get_world_scale():
            return torch.ones(3, dtype=torch.float32)

    link_a = Link([1.0, 2.0, 0.0], math.radians(90.0))
    link_b = Link([-1.0, 0.5, 0.0], 0.0)
    meshes = [
        Mesh("/mesh/a0", link_a, [0.1, 0.0, 0.0]),
        Mesh("/mesh/a1", link_a, [0.0, 0.2, 0.0]),
        Mesh("/mesh/b0", link_b, [0.3, 0.0, 0.0]),
        Mesh("/mesh/b1", link_b, [0.0, -0.4, 0.0]),
    ]
    link_a.collision_meshes = {
        mesh.prim_path: mesh for mesh in meshes if mesh.link is link_a
    }
    link_b.collision_meshes = {
        mesh.prim_path: mesh for mesh in meshes if mesh.link is link_b
    }
    class SceneObject:
        visual_only = False
        links = {"a": link_a, "b": link_b}

    obj = SceneObject()
    root = SimpleNamespace(
        get_position_orientation=lambda: (
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
    )
    robot = SimpleNamespace(root_link=root)
    robot.scene = SimpleNamespace(objects=[robot, obj])
    generator = SimpleNamespace(robot=robot)

    first = RealCuroboBackend._current_collision_mesh_snapshot(
        generator,
        full_digest=True,
    )
    # One parent-link read is for the grouped cache and one per mesh comes
    # from the fake mesh's initial authoritative world-pose read.
    assert link_a.pose_reads == 3
    assert link_b.pose_reads == 3
    assert all(mesh.pose_reads == 1 for mesh in meshes)
    assert first["snapshot_timings"]["unique_link_count"] == 2

    link_a.position = torch.tensor([2.0, -1.0, 0.0])
    link_a.yaw = math.radians(-90.0)
    link_b.position = torch.tensor([0.5, 1.5, 0.0])
    second = RealCuroboBackend._current_collision_mesh_snapshot(
        generator,
        full_digest=False,
        kinematic_cache=first["kinematic_cache"],
    )

    assert link_a.pose_reads == 4
    assert link_b.pose_reads == 4
    assert all(mesh.pose_reads == 1 for mesh in meshes)
    assert second["snapshot_timings"]["unique_link_count"] == 2
    poses = dict(second["poses"])
    for mesh in meshes:
        expected_position, expected_xyzw = mat2pose(
            pose2mat(mesh.link.get_position_orientation()) @ mesh.local
        )
        expected = [
            *expected_position.tolist(),
            float(expected_xyzw[3]),
            *expected_xyzw[:3].tolist(),
        ]
        np.testing.assert_allclose(poses[mesh.prim_path], expected, atol=1e-6)


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


def test_real_warmup_includes_base_and_whole_body_identity_plans_without_actions(
    tmp_path,
):
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
    base_calls = []

    def base_plan(**kwargs):
        base_calls.append(kwargs)
        return {
            "ok": True,
            "metrics": {
                "trajectory_waypoints": 1,
                "collision_admission": {"admitted": True},
            },
        }

    backend._compute_base_plan = base_plan
    arm_calls = []

    def arm_plan(**kwargs):
        arm_calls.append(kwargs)
        return {"ok": True}

    whole_body_calls = []
    backend.plan_whole_body_trajectory = lambda **kwargs: (
        whole_body_calls.append(kwargs) or {"ok": True}
    )
    feedback_warmup = {
        "query": "read_only_post_step_safety_feedback",
        "ok": True,
        "env_actions_sent": 0,
        "simulator_advanced": False,
    }
    backend._warmup_dashboard_feedback = lambda: feedback_warmup

    result = backend.warmup()

    assert result["status"] == "complete"
    assert len(base_calls) == 1
    np.testing.assert_allclose(base_calls[0]["target_xyyaw"], [0.0, 0.0, 0.0])
    assert base_calls[0]["timeout_s"] == pytest.approx(120.0)
    assert base_calls[0]["skip_obstacle_update"] is False
    assert base_calls[0]["attempt_timeout_cap_s"] == pytest.approx(8.0)
    assert base_calls[0]["solver_timeout_cap_s"] == pytest.approx(6.0)
    assert base_calls[0]["planning_profile"] == RESET_IDENTITY_WARMUP_PROFILE
    assert base_calls[0]["wall_clock_timeout_s"] == pytest.approx(120.0)
    assert arm_calls == []
    assert [call["hand"] for call in whole_body_calls] == ["left", "right"]
    for call in whole_body_calls:
        expected_position, expected_quaternion = eef_poses[call["hand"]]
        np.testing.assert_allclose(call["target_xyz"], expected_position)
        np.testing.assert_allclose(
            call["target_quat_xyzw"],
            expected_quaternion,
        )
        assert call["timeout_s"] == pytest.approx(12.0)
        assert call["search_profile"] == RESET_IDENTITY_WARMUP_PROFILE
        assert call["search_profile"] != WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
        assert call["wall_clock_timeout_s"] == pytest.approx(120.0)
    assert result["identity_warmup"]["env_actions_sent"] == 0
    assert result["identity_warmup"]["simulator_advanced"] is False
    assert result["identity_warmup"]["base"] == {
        "query": "current_pose_identity_trajectory",
        "ok": True,
        "stop_reason": None,
        "trajectory_waypoints": 1,
        "collision_admitted": True,
    }
    assert [sample["query"] for sample in result["identity_warmup"]["hands"]] == [
        "identity_trajectory",
        "identity_trajectory",
    ]
    assert result["identity_warmup"]["dashboard_feedback"] == feedback_warmup
    artifact = Path(result["artifact"])
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == "complete"


def test_dashboard_feedback_warmup_compiles_all_read_only_safety_paths_without_action(
    tmp_path,
):
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    current_q = np.zeros(28, dtype=np.float32)
    calls: list[str] = []
    backend.get_joint_positions = lambda: current_q.copy()
    backend.get_attached_object = lambda hand: (
        calls.append(f"attachment:{hand}") or None
    )
    backend.capture_navigation_isolation_reference = lambda: (
        calls.append("isolation_reference")
        or {
            "mode": "base_only",
            "gripper_commands": {"left": 1.0, "right": -1.0},
        }
    )
    def isolation_report(**kwargs):
        calls.append("isolation_report")
        action = np.asarray(kwargs["action"])
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["left_gripper"]],
            1.0,
        )
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["right_gripper"]],
            -1.0,
        )
        return {"available": True, "ok": True}

    backend.navigation_isolation_report = isolation_report
    backend.capture_whole_body_contact_baseline = lambda **_kwargs: (
        calls.append("contact_baseline") or {"available": True}
    )
    backend.whole_body_contact_report = lambda **_kwargs: (
        calls.append("contact_report")
        or {"available": True, "unexpected_contact": False}
    )
    backend.joint_tracking_report = lambda *_args, **_kwargs: (
        calls.append("tracking_report")
        or {
            "available": True,
            "max_base_xy_error_m": 0.0,
            "base_yaw_error_rad": 0.0,
            "max_articulation_error_rad": 0.0,
        }
    )
    backend.get_base_pose = lambda: (
        calls.append("base_pose") or np.zeros(3, dtype=np.float64)
    )
    backend._step_env_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("read-only warmup must not send an env action")
    )
    backend.joint_target_to_action = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(
            AssertionError("read-only warmup must not install an action adapter")
        )
    )

    result = backend._warmup_dashboard_feedback()

    assert result["ok"] is True
    assert result["env_actions_sent"] == 0
    assert result["simulator_advanced"] is False
    assert result["checks"] and all(result["checks"].values())
    assert calls == [
        "attachment:left",
        "attachment:right",
        "isolation_reference",
        "isolation_report",
        "contact_baseline",
        "contact_report",
        "tracking_report",
        "base_pose",
    ]


def test_background_whole_body_planning_does_not_enter_signal_deadline(
    monkeypatch,
):
    backend = RealCuroboBackend(None)
    backend._compute_whole_body_plan = lambda **_kwargs: {
        "ok": True,
        "metrics": {},
    }

    def forbidden_deadline(*_args, **_kwargs):
        raise AssertionError("background planning must not install SIGALRM")

    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        forbidden_deadline,
    )
    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.zeros(3),
        target_quat_xyzw=None,
        timeout_s=12.0,
        search_profile=WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
        background=True,
    )

    assert result["ok"] is True
    assert result["metrics"]["deadline_enforcement"] == {
        "solver_timeout_enforced": True,
        "hard_wall_clock_enforced": False,
        "hard_wall_clock_deadline_s": None,
        "soft_deadline_s": 12.0,
        "solver_planning_budget_s": 12.0,
    }


def test_reset_warmup_wall_override_preserves_dashboard_solver_budget(
    monkeypatch,
):
    deadlines = []
    compute_calls = []

    @contextmanager
    def recording_deadline(timeout_s, operation):
        deadlines.append((str(operation), float(timeout_s)))
        yield

    backend = RealCuroboBackend(None)
    backend._compute_whole_body_plan = lambda **kwargs: (
        compute_calls.append(kwargs) or {"ok": True, "metrics": {}}
    )
    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        recording_deadline,
    )
    dashboard = {
        "hand": "left",
        "target_xyz": np.zeros(3),
        "target_quat_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "timeout_s": 12.0,
        "search_profile": WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
    }
    warmup = {
        **dashboard,
        "search_profile": RESET_IDENTITY_WARMUP_PROFILE,
    }

    runtime = backend.plan_whole_body_trajectory(**dashboard)
    reset_warmup = backend.plan_whole_body_trajectory(
        **warmup,
        wall_clock_timeout_s=RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
    )

    assert runtime["ok"] is True
    assert reset_warmup["ok"] is True
    assert [call["timeout_s"] for call in compute_calls] == pytest.approx(
        [12.0, 12.0]
    )
    assert [call["search_profile"] for call in compute_calls] == [
        WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
        RESET_IDENTITY_WARMUP_PROFILE,
    ]
    assert deadlines == [
        ("left whole-body planning transaction", pytest.approx(12.0)),
        (
            "left whole-body planning transaction",
            pytest.approx(RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S),
        ),
    ]
    assert runtime["metrics"]["deadline_enforcement"][
        "solver_planning_budget_s"
    ] == pytest.approx(12.0)
    assert reset_warmup["metrics"]["deadline_enforcement"][
        "solver_planning_budget_s"
    ] == pytest.approx(12.0)


def test_reset_identity_warmup_profile_excludes_dashboard_local_ik():
    profile = _whole_body_search_profile(RESET_IDENTITY_WARMUP_PROFILE)

    assert profile["name"] == RESET_IDENTITY_WARMUP_PROFILE
    assert profile["name"] != WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
    assert "local_ik_deadline_s" not in profile
    assert profile["planning_deadline_s"] == pytest.approx(12.0)
    assert profile["fast_trajopt_deadline_s"] == pytest.approx(4.0)


def test_real_warmup_base_identity_failure_is_fail_closed_and_artifacted(
    tmp_path,
):
    class Robot:
        @staticmethod
        def get_joint_positions():
            return np.zeros(28, dtype=np.float32)

    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._find_robot = lambda: Robot()
    backend._base_xy_yaw = lambda _robot: np.zeros(4)
    backend._compute_base_plan = lambda **_kwargs: {
        "ok": False,
        "stop_reason": "collision",
        "metrics": {"env_actions_sent": 0},
    }
    backend.plan_whole_body_trajectory = lambda **_kwargs: pytest.fail(
        "hand warmup must not run after BASE identity failure"
    )

    with pytest.raises(
        RuntimeError,
        match="BASE current-pose cuRobo warmup failed closed",
    ):
        backend.warmup()

    artifact = tmp_path / "planner_curobo_warmup.json"
    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["artifact"] == str(artifact)
    assert report["identity_warmup"]["env_actions_sent"] == 0
    assert report["identity_warmup"]["simulator_advanced"] is False


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
def test_base_plan_requires_dense_collision_admission(tmp_path, monkeypatch):
    wall_clock_calls = []

    @contextmanager
    def recording_deadline(timeout_s, operation):
        wall_clock_calls.append((str(operation), float(timeout_s)))
        yield

    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        recording_deadline,
    )
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
            self.compute_calls = 0
            self.collision_q = None
            self.collision_kwargs = []

        def compute_trajectories(self, _positions, _quaternions, **kwargs):
            self.compute_calls += 1
            self.compute_kwargs = kwargs
            return np.array([True, False]), [path, None]

        @staticmethod
        def path_to_joint_trajectory(_path, **_kwargs):
            raise AssertionError("BASE full-js augmentation must not be called")

        def check_collisions(self, q, **kwargs):
            assert kwargs["self_collision_check"] is True
            self.collision_q = np.asarray(q).copy()
            self.collision_kwargs.append(dict(kwargs))
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
        timeout_s=12.0,
    )

    assert result["ok"] is True
    assert generator.compute_calls == 1
    assert generator.collision_kwargs[-1]["skip_obstacle_update"] is True
    assert generator.compute_kwargs["ik_only"] is True
    assert generator.compute_kwargs["ik_world_collision_check"] is True
    assert generator.compute_kwargs["skip_obstacle_update"] is False
    assert generator.compute_kwargs["timeout"] == pytest.approx(6.0)
    assert result["metrics"]["attempt_timeout_budget_s"] == pytest.approx(8.0)
    assert result["metrics"]["attempt_timeout_s"] == pytest.approx(6.0)
    assert result["metrics"]["solver_timeout_s"] == pytest.approx(6.0)
    assert result["metrics"]["attempt_timeout_cap_s"] == pytest.approx(8.0)
    assert result["metrics"]["solver_timeout_cap_s"] == pytest.approx(6.0)
    assert result["metrics"]["planning_profile"] == "default"
    assert result["metrics"]["collision_admission_enabled"] is True
    assert result["metrics"]["obstacle_update"] is True
    assert result["metrics"]["collision_admission"]["admitted"] is True
    assert result["metrics"]["base_trajectory_certificate"][
        "post_interpolation_check"
    ] is True
    trajectory = np.asarray(result["joint_trajectory"], dtype=np.float32)
    certificate = result["metrics"]["base_trajectory_certificate"]
    assert len(trajectory) >= 1
    assert certificate["waypoint_count"] == len(trajectory)
    assert certificate["trajectory_sha256"] == hashlib.sha256(
        np.ascontiguousarray(trajectory).tobytes()
    ).hexdigest()
    assert generator.collision_q is not None
    assert len(generator.collision_q) > len(trajectory)
    resampling = result["metrics"]["execution_resampling"]
    assert resampling["source"] == "base_ik_minimum_jerk_execution_resampling"
    assert resampling["method"] == "quintic_minimum_jerk"
    assert resampling["measured_max_xy_step_m"] <= (
        BASE_EXECUTION_XY_STEP_M + 1e-9
    )
    assert resampling["measured_max_yaw_step_rad"] <= (
        BASE_EXECUTION_YAW_STEP_RAD + 1e-9
    )

    prepared_result = backend._compute_base_plan(
        target_xyyaw=np.array([0.2, 0.1, 0.2]),
        timeout_s=12.0,
        attempt_timeout_cap_s=4.0,
        solver_timeout_cap_s=4.0,
        planning_profile=DASHBOARD_PREPARED_BASE_PLANNING_PROFILE,
    )

    assert prepared_result["ok"] is True
    assert generator.compute_calls == 1
    assert generator.collision_kwargs[-1]["skip_obstacle_update"] is False
    assert prepared_result["metrics"]["solver_invoked"] is False
    assert prepared_result["metrics"]["base_goal_construction"] == {
        "method": "analytic_full_q_base_assignment",
        "active_base_indices": [0, 1, 5],
        "nonbase_locked_to_call_start": True,
        "continuous_shortest_yaw_arc": True,
    }
    assert prepared_result["metrics"][
        "attempt_timeout_budget_s"
    ] == pytest.approx(
        4.0
    )
    assert prepared_result["metrics"]["attempt_timeout_s"] == pytest.approx(4.0)
    assert prepared_result["metrics"]["solver_timeout_s"] == pytest.approx(4.0)
    assert prepared_result["metrics"]["attempt_timeout_cap_s"] == pytest.approx(
        4.0
    )
    assert prepared_result["metrics"]["solver_timeout_cap_s"] == pytest.approx(
        4.0
    )
    assert prepared_result["metrics"]["planning_profile"] == (
        DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
    )
    assert prepared_result["metrics"]["execution_resampling"]["source"] == (
        "analytic_full_q_minimum_jerk_execution_resampling"
    )
    assert prepared_result["metrics"]["execution_resampling"][
        "xy_step_limit_m"
    ] == pytest.approx(DASHBOARD_BASE_EXECUTION_XY_STEP_M)
    assert prepared_result["metrics"]["execution_resampling"][
        "yaw_step_limit_rad"
    ] == pytest.approx(DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD)
    prepared_trajectory = np.asarray(
        prepared_result["joint_trajectory"], dtype=np.float32
    )
    np.testing.assert_array_equal(prepared_trajectory[:, 2:5], 0.0)
    np.testing.assert_array_equal(prepared_trajectory[:, 6:], 0.0)

    warmup_result = backend._compute_base_plan(
        target_xyyaw=np.array([0.2, 0.1, 0.2]),
        timeout_s=RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
        attempt_timeout_cap_s=8.0,
        solver_timeout_cap_s=6.0,
        planning_profile=RESET_IDENTITY_WARMUP_PROFILE,
        wall_clock_timeout_s=RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
    )

    assert warmup_result["ok"] is True
    assert generator.compute_calls == 2
    assert generator.compute_kwargs["timeout"] == pytest.approx(6.0)
    assert warmup_result["metrics"]["attempt_timeout_budget_s"] == pytest.approx(
        8.0
    )
    assert warmup_result["metrics"]["solver_timeout_s"] == pytest.approx(6.0)
    assert warmup_result["metrics"]["planning_profile"] == (
        RESET_IDENTITY_WARMUP_PROFILE
    )
    assert warmup_result["metrics"]["deadline_enforcement"][
        "hard_wall_clock_deadline_s"
    ] == pytest.approx(RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S)
    assert wall_clock_calls == [
        ("BASE cuRobo candidate", pytest.approx(8.0)),
        ("BASE cuRobo candidate", pytest.approx(4.0)),
        (
            "BASE cuRobo candidate",
            pytest.approx(RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S),
        ),
    ]


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


def test_whole_body_tracking_l2_wraps_equivalent_two_pi_base_yaw():
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
            current[5] = math.pi - 0.01
            return current

    backend = RealCuroboBackend(None)
    backend._robot = Robot()
    target = Robot.get_joint_positions()
    target[5] -= np.float32(2.0 * math.pi)
    target[6] = 0.003

    report = backend.joint_tracking_report(target, hand=None)

    assert report["available"] is True
    assert report["reached"] is True
    assert report["base_yaw_error_rad"] <= 1e-6
    assert report["normalized_21d_tracking_error"] == pytest.approx(0.15)
    assert report["active_joint_l2_error"] == pytest.approx(0.003, abs=1e-6)


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

    result = _move_to_without_replan_checkpoint(
        executor,
        hand=hand,
        target_xyz=[0.55, 0.0, 0.0],
    )

    assert result["primitive_success"] is True
    assert result["metrics"]["motion_scope"] == "whole_body"
    assert result["metrics"]["generator_kind"] == "whole_body"
    assert result["metrics"]["active_dof_count"] == 21
    assert result["metrics"]["whole_body_eef_path_guard"][
        "waypoint_settle_position_tolerance_m"
    ] == pytest.approx(0.002)
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


def test_whole_body_live_eef_divergence_stops_before_next_action():
    class Backend(_FakeBackend):
        def advance(self):
            assert self.target is not None
            self.pose = self.target + np.asarray([0.2, 0.0, 0.0])

    backend = Backend()
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="right",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert result["metrics"]["executed_waypoints"] == 1
    assert result["metrics"]["partial_motion"] is True
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert len(env.calls) == 1
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["live_waypoint_position_tolerance_m"] == pytest.approx(0.005)
    assert guard["live_waypoint_orientation_tolerance_rad"] == pytest.approx(
        math.radians(1.0)
    )
    assert guard["last_waypoint"]["violations"]["waypoint_position"] is True
    assert guard["max_observed_live_waypoint_position_error_m"] > 0.005


def test_whole_body_streams_each_intermediate_waypoint_exactly_once():
    class Backend(_FakeBackend):
        marker_step = 0.0001
        delayed_waypoint = 3

        def __init__(self):
            super().__init__()
            self.commanded_waypoint_indices = []
            self.waypoint_command_counts = {}

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(len(trajectory), dtype=np.float32) * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            return self._refresh_whole_body_certificate(result)

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            target = np.asarray(target_q, dtype=np.float32).reshape(-1)
            waypoint_index = int(round(float(target[6]) / self.marker_step))
            self.commanded_waypoint_indices.append(waypoint_index)
            count = self.waypoint_command_counts.get(waypoint_index, 0) + 1
            self.waypoint_command_counts[waypoint_index] = count
            nominal = self.execution_eef_positions[waypoint_index].astype(
                np.float64
            )
            if waypoint_index == self.delayed_waypoint and count == 1:
                nominal = nominal - np.asarray([0.0, 0.0044, 0.0])
            self.next_execution_eef_pose = nominal
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                waypoint_index
            ].astype(np.float64)
            return action

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is True, result.get("stop_reason")
    assert backend.commanded_waypoint_indices[3:5] == [3, 4]
    assert all(
        count == 1
        for waypoint, count in backend.waypoint_command_counts.items()
        if waypoint < max(backend.commanded_waypoint_indices)
    )
    assert len(env.calls) == 20
    delayed = [
        row["whole_body_eef_waypoint_tracking"]
        for row in result["trace"]
        if row.get("whole_body_eef_waypoint_tracking", {}).get(
            "commanded_index"
        )
        == 3
    ]
    assert len(delayed) == 1
    assert delayed[0]["waypoint_settled"] is False
    assert delayed[0]["post_step_index"] == 4
    assert delayed[0]["joint_waypoint_reached"] is True
    assert delayed[0]["position_error_m"] == pytest.approx(0.0044, abs=1e-6)
    assert delayed[0][
        "configured_position_settle_tolerance_m"
    ] == pytest.approx(0.004)
    assert delayed[0][
        "effective_position_settle_tolerance_m"
    ] < 0.004
    assert delayed[0]["prospective_position_bound_m"] > 0.005
    assert delayed[0]["violations"]["waypoint_position"] is False
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["waypoint_settle_position_tolerance_m"] == pytest.approx(
        0.004
    )
    certificate = result["metrics"]["whole_body_certificate"]
    assert certificate["active_dof_count"] == 21
    assert certificate["inactive_eef_goal_count"] == 0
    assert certificate["attachment_hand_count"] == 2


@pytest.mark.parametrize("success_on_terminal_command", [6, None])
def test_whole_body_terminal_has_six_command_total_cap_without_seventh(
    success_on_terminal_command,
):
    class Backend(_FakeBackend):
        marker_step = 0.0001

        def __init__(self):
            super().__init__()
            self.commanded_waypoint_indices = []
            self.terminal_commands = 0

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(len(trajectory), dtype=np.float32) * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            return self._refresh_whole_body_certificate(result)

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            target = np.asarray(target_q, dtype=np.float32).reshape(-1)
            waypoint_index = int(round(float(target[6]) / self.marker_step))
            self.commanded_waypoint_indices.append(waypoint_index)
            terminal_index = len(self.execution_eef_positions) - 1
            if waypoint_index == terminal_index:
                self.terminal_commands += 1
                if (
                    success_on_terminal_command is None
                    or self.terminal_commands < success_on_terminal_command
                ):
                    self.next_execution_eef_pose = (
                        self.execution_eef_positions[-1].astype(np.float64)
                        - np.asarray([0.002, 0.0, 0.0])
                    )
            return action

    backend = Backend()
    executor, env = _executor(backend)
    target = np.asarray([0.53, 0.0, 0.0], dtype=np.float64)
    plan = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        timeout_s=12.0,
        attached_obj=None,
    )

    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD,
        timeout_s=12.0,
        require_pose=True,
        hold_steps_required=0,
        joint_trajectory=plan["joint_trajectory"],
        expected_attachments_by_hand=plan["expected_attachments_by_hand"],
        motion_scope="whole_body",
        whole_body_certificate=plan["whole_body_certificate"],
    )

    terminal_index = len(backend.execution_eef_positions) - 1
    assert backend.terminal_commands == TERMINAL_COMMAND_LIMIT
    assert backend.commanded_waypoint_indices.count(terminal_index) == (
        TERMINAL_COMMAND_LIMIT
    )
    assert len(env.calls) == terminal_index + TERMINAL_COMMAND_LIMIT
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["terminal_commands_sent"] == TERMINAL_COMMAND_LIMIT
    if success_on_terminal_command is None:
        assert result["primitive_success"] is False
        assert result["stop_reason"] == "target_tolerance_not_met"
    else:
        assert result["primitive_success"] is True
        assert result["stop_reason"] == "reached"


def test_whole_body_runtime_never_reads_dynamics_reporter():
    class Backend(_FakeBackend):
        @staticmethod
        def dynamics_report():
            raise AssertionError("whole-body runtime must not read dynamics")

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is True, result.get("stop_reason")
    assert env.calls


def test_whole_body_unreached_intermediate_is_not_repeated_below_hard_threshold():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.tracking_calls = 0
            self.runtime_q_targets = []

        def dynamics_report(self):
            raise AssertionError("whole-body runtime must not read dynamics")

        def joint_tracking_report(self, target_q, *, hand):
            self.tracking_calls += 1
            report = super().joint_tracking_report(target_q, hand=hand)
            if self.tracking_calls == 1:
                report["reached"] = False
                report["max_articulation_error_rad"] = 0.1
            return report

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            self.runtime_q_targets.append(
                np.asarray(target_q, dtype=np.float32).copy()
            )
            return super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is True
    assert len(env.calls) == 20
    assert backend.tracking_calls == len(env.calls)
    commanded_indices = [
        row["whole_body_eef_waypoint_tracking"]["commanded_index"]
        for row in result["trace"]
        if row.get("whole_body_eef_waypoint_tracking") is not None
    ]
    assert commanded_indices[:2] == [0, 1]


class _InitialAxisReverseBackend(_FakeBackend):
    marker_step = 0.0001

    def __init__(
        self,
        first_waypoint_feedback=None,
        *,
        waypoint_feedback=None,
        moving_dynamics_calls=(2,),
    ):
        super().__init__()
        self.waypoint_feedback = dict(waypoint_feedback or {})
        for attempt, feedback in enumerate(
            first_waypoint_feedback or (),
            start=1,
        ):
            self.waypoint_feedback[(0, attempt)] = feedback
        self.moving_dynamics_calls = set(moving_dynamics_calls)
        self.call_start_pose = self.pose.copy()
        self.commanded_waypoint_indices = []
        self.commanded_q = []
        self.waypoint_command_counts = {}
        self.dynamics_calls = 0

    def plan_whole_body_trajectory(self, **kwargs):
        result = super().plan_whole_body_trajectory(**kwargs)
        trajectory = np.asarray(
            result["joint_trajectory"], dtype=np.float32
        ).copy()
        trajectory[:, 6] = (
            np.arange(len(trajectory), dtype=np.float32) * self.marker_step
        )
        result["joint_trajectory"] = trajectory
        return self._refresh_whole_body_certificate(result)

    def dynamics_report(self):
        self.dynamics_calls += 1
        return _grouped_dynamics_report(
            base_translation=(
                0.01 if self.dynamics_calls in self.moving_dynamics_calls else 0.0
            )
        )

    def joint_target_to_action(
        self,
        target_q,
        *,
        hand,
        fixed_reference=None,
    ):
        action = super().joint_target_to_action(
            target_q,
            hand=hand,
            fixed_reference=fixed_reference,
        )
        target = np.asarray(target_q, dtype=np.float32).reshape(-1)
        waypoint_index = int(round(float(target[6]) / self.marker_step))
        self.commanded_waypoint_indices.append(waypoint_index)
        self.commanded_q.append(target.copy())
        count = self.waypoint_command_counts.get(waypoint_index, 0) + 1
        self.waypoint_command_counts[waypoint_index] = count
        feedback = self.waypoint_feedback.get((waypoint_index, count))
        if feedback is not None:
            along_m, lateral_m = feedback
            self.next_execution_eef_pose = self.call_start_pose + np.asarray(
                [along_m, lateral_m, 0.0],
                dtype=np.float64,
            )
        else:
            self.next_execution_eef_pose = self.execution_eef_positions[
                waypoint_index
            ].astype(np.float64)
        self.next_execution_eef_quat = self.execution_eef_quaternions[
            waypoint_index
        ].astype(np.float64)
        return action


def test_initial_axis_reverse_with_other_violation_hard_fails():
    backend = _InitialAxisReverseBackend([(-0.0016345, 0.0051)])
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert len(env.calls) == 1
    waypoint = result["metrics"]["whole_body_eef_path_guard"]["last_waypoint"]
    assert waypoint["violations"]["axis_monotonic"] is True
    assert waypoint["violations"]["lateral"] is True
    assert waypoint["fatal_violations"]["lateral"] is True
    assert waypoint["initial_axis_reverse_deferred"] is False
    assert waypoint["deferred_violations"] == {}
    assert len(
        result["metrics"]["whole_body_eef_path_guard"][
            "axis_reverse_defer_ledger"
        ]
    ) == 0
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_first_axis_reverse_hard_fails_without_repeating_waypoint():
    backend = _InitialAxisReverseBackend(
        [(-0.0015, 0.0), (-0.0026, 0.0)]
    )
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert len(env.calls) == 1
    assert backend.commanded_waypoint_indices == [0]
    waypoint = result["metrics"]["whole_body_eef_path_guard"]["last_waypoint"]
    assert waypoint["reverse_step_m"] == pytest.approx(0.0015)
    assert waypoint["violations"]["axis_monotonic"] is True
    assert waypoint["fatal_violations"]["axis_monotonic"] is True
    assert waypoint["initial_axis_reverse_deferred"] is False
    assert waypoint["deferred_violations"] == {}
    assert result["metrics"]["whole_body_eef_path_guard"][
        "axis_reverse_defer_ledger"
    ] == {}
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_initial_axis_reverse_above_transient_cap_hard_fails():
    reverse_m = WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M + 1e-6
    backend = _InitialAxisReverseBackend([(-reverse_m, 0.0)])
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert len(env.calls) == 1
    waypoint = result["metrics"]["whole_body_eef_path_guard"]["last_waypoint"]
    assert waypoint["reverse_step_m"] == pytest.approx(reverse_m)
    assert waypoint["violations"]["axis_monotonic"] is True
    assert waypoint["fatal_violations"]["axis_monotonic"] is True
    assert waypoint["initial_axis_reverse_deferred"] is False
    assert waypoint["deferred_violations"] == {}
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_initial_axis_reverse_contact_path_never_defers():
    backend = _InitialAxisReverseBackend([(-0.0016345, 0.0)])
    executor, env = _executor(backend)

    result = executor._move_to_composite_stage(
        hand="left",
        target_xyz=np.asarray([0.53, 0.0, 0.0]),
        target_quat_xyzw=None,
        position_tolerance_m=0.002,
        orientation_tolerance_rad=math.radians(5.0),
        timeout_s=10.0,
        hold_steps_required=1,
        contact_target_xyz=np.asarray([0.53, 0.0, 0.0]),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert len(env.calls) == 1
    waypoint = result["metrics"]["whole_body_eef_path_guard"]["last_waypoint"]
    assert waypoint["violations"]["axis_monotonic"] is True
    assert waypoint["fatal_violations"]["axis_monotonic"] is True
    assert waypoint["initial_axis_reverse_deferred"] is False
    assert waypoint["deferred_violations"] == {}
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_initial_axis_reverse_attached_object_path_never_defers():
    backend = _InitialAxisReverseBackend([(-0.0016345, 0.0)])
    backend.attached_obj = {"left_eef_link": backend.target_root}
    executor, env = _executor(backend)
    env._gripper_latch["left"] = -1.0

    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert len(env.calls) == 1
    guard = result["metrics"]["whole_body_eef_path_guard"]
    waypoint = guard["last_waypoint"]
    assert waypoint["violations"]["axis_monotonic"] is True
    assert waypoint["fatal_violations"]["axis_monotonic"] is True
    assert waypoint["axis_reverse_deferred"] is False
    assert guard["axis_reverse_defer_ledger"] == {}
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_prepared_dashboard_eef_defers_retry10_1191um_first_sample_once():
    class Backend(_InitialAxisReverseBackend):
        def plan_whole_body_trajectory(
            self,
            *,
            start_q=None,
            start_eef_pose=None,
            background=False,
            **kwargs,
        ):
            del start_q, start_eef_pose, background
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend([(-0.001191, 0.0)])
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("left_arm", "forward")

    assert prepared["execution_policy"] == (
        PREPARED_DASHBOARD_EEF_EXECUTION_POLICY
    )
    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "retry10-left-arm-forward",
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert backend.commanded_waypoint_indices[:3] == [0, 0, 1]
    assert np.array_equal(backend.commanded_q[0], backend.commanded_q[1])
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["execution_policy"] == (
        PREPARED_DASHBOARD_EEF_EXECUTION_POLICY
    )
    assert guard["first_sample_axis_reverse_transient_enabled"] is True
    assert len(guard["axis_reverse_defer_ledger"]) == 1
    deferred = next(iter(guard["axis_reverse_defer_ledger"].values()))
    assert deferred["reverse_step_m"] == pytest.approx(0.001191)
    assert deferred["maximum_reverse_step_m"] == pytest.approx(
        WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M
    )
    assert deferred["exact_same_q_repeat_required"] is True
    assert guard["pending_axis_reverse_settle_repeat"] is None
    first_sample = next(
        row["whole_body_eef_waypoint_tracking"]
        for row in result["trace"]
        if (
            isinstance(row.get("whole_body_eef_waypoint_tracking"), dict)
            and row["whole_body_eef_waypoint_tracking"].get(
                "initial_axis_reverse_deferred"
            )
            is True
        )
    )
    assert first_sample["violations"]["axis_monotonic"] is True
    assert first_sample["fatal_violations"]["axis_monotonic"] is False
    assert first_sample["deferred_violations"] == {
        "axis_monotonic": (
            "prepared_dashboard_first_sample_controller_transient"
        )
    }
    assert len(env.calls) == len(backend.execution_eef_positions) + 1


def test_prepared_dashboard_eef_repeat_axis_reverse_is_fatal_without_third_action():
    class Backend(_InitialAxisReverseBackend):
        def plan_whole_body_trajectory(
            self,
            *,
            start_q=None,
            start_eef_pose=None,
            background=False,
            **kwargs,
        ):
            del start_q, start_eef_pose, background
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend(
        waypoint_feedback={
            (0, 1): (-0.001191, 0.0),
            (0, 2): (-0.0025, 0.0),
        }
    )
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("left_arm", "forward")

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "retry10-repeat-reverse",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert backend.commanded_waypoint_indices == [0, 0]
    assert len(env.calls) == 2
    waypoint = result["metrics"]["whole_body_eef_path_guard"]["last_waypoint"]
    assert waypoint["axis_reverse_settle_repeat_command"] is True
    assert waypoint["fatal_violations"]["axis_monotonic"] is True
    assert waypoint["axis_reverse_deferred"] is False
    assert result["metrics"]["post_stop_env_actions"] == 0


@pytest.mark.parametrize(
    ("position_error_m", "expected_stop_reason"),
    [
        (0.002498, None),
        (0.002500, None),
        (0.005001, "eef_path_divergence"),
    ],
)
def test_short_jog_streams_once_and_keeps_hard_corridor_boundary(
    position_error_m,
    expected_stop_reason,
):
    class Backend(_FakeBackend):
        marker_step = 0.0001

        def __init__(self):
            super().__init__()
            self.waypoint_command_counts = {}

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(len(trajectory), dtype=np.float32) * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            return self._refresh_whole_body_certificate(result)

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            target = np.asarray(target_q, dtype=np.float32).reshape(-1)
            waypoint_index = int(round(float(target[6]) / self.marker_step))
            count = self.waypoint_command_counts.get(waypoint_index, 0) + 1
            self.waypoint_command_counts[waypoint_index] = count
            nominal = self.execution_eef_positions[waypoint_index].astype(
                np.float64
            )
            if waypoint_index == 0 and count == 1:
                nominal = nominal - np.asarray(
                    [0.0, position_error_m, 0.0]
                )
            self.next_execution_eef_pose = nominal
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                waypoint_index
            ].astype(np.float64)
            return action

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert backend.waypoint_command_counts[0] == 1
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["waypoint_settle_position_tolerance_m"] == pytest.approx(
        0.004
    )
    first_waypoint = (
        next(
            row["whole_body_eef_waypoint_tracking"]
            for row in result["diagnostics"]["trace"]
            if row.get("whole_body_eef_waypoint_tracking", {}).get(
                "commanded_index"
            )
            == 0
        )
        if expected_stop_reason is None
        else guard["last_waypoint"]
    )
    assert first_waypoint[
        "configured_position_settle_tolerance_m"
    ] == pytest.approx(0.004)
    assert first_waypoint[
        "prospective_next_waypoint_step_m"
    ] == pytest.approx(0.0015, abs=1e-6)
    assert first_waypoint[
        "effective_position_settle_tolerance_m"
    ] == pytest.approx(0.002499, abs=1e-6)
    if expected_stop_reason is None:
        assert result["primitive_success"] is True, result.get("stop_reason")
        assert len(env.calls) == 20
    else:
        assert result["primitive_success"] is False
        assert result["stop_reason"] == expected_stop_reason
        assert len(env.calls) == 1
        assert guard["last_waypoint"]["violations"]["lateral"] is True


def test_whole_body_streams_once_with_subthreshold_orientation_lag():
    class Backend(_FakeBackend):
        marker_step = 0.0001
        delayed_waypoint = 3

        def __init__(self):
            super().__init__()
            self.commanded_waypoint_indices = []
            self.waypoint_command_counts = {}

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(len(trajectory), dtype=np.float32) * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            angles = np.deg2rad(
                np.minimum(np.arange(1, len(trajectory) + 1), 6)
            )
            quaternions = np.column_stack(
                [
                    np.zeros(len(angles)),
                    np.zeros(len(angles)),
                    np.sin(angles * 0.5),
                    np.cos(angles * 0.5),
                ]
            ).astype(np.float32)
            self.execution_eef_quaternions = quaternions
            return self._refresh_whole_body_certificate(result)

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            target = np.asarray(target_q, dtype=np.float32).reshape(-1)
            waypoint_index = int(round(float(target[6]) / self.marker_step))
            self.commanded_waypoint_indices.append(waypoint_index)
            count = self.waypoint_command_counts.get(waypoint_index, 0) + 1
            self.waypoint_command_counts[waypoint_index] = count
            self.next_execution_eef_pose = self.execution_eef_positions[
                waypoint_index
            ].astype(np.float64)
            quat = self.execution_eef_quaternions[waypoint_index].astype(
                np.float64
            )
            if waypoint_index >= self.delayed_waypoint and count == 1:
                angle = math.radians(
                    min(waypoint_index + 1, 6) - 0.6
                )
                quat = np.asarray(
                    [0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5)]
                )
            self.next_execution_eef_quat = quat
            return action

    backend = Backend()
    executor, env = _executor(backend)
    target_angle = math.radians(6.0)

    result = executor.move_to(
        hand="left",
        target_xyz=backend.pose.copy(),
        target_quat_xyzw=[
            0.0,
            0.0,
            math.sin(target_angle * 0.5),
            math.cos(target_angle * 0.5),
        ],
    )

    assert result["primitive_success"] is True, result.get("stop_reason")
    assert backend.commanded_waypoint_indices[3:5] == [3, 4]
    assert len(env.calls) == 20
    delayed = [
        row["whole_body_eef_waypoint_tracking"]
        for row in result["trace"]
        if row.get("whole_body_eef_waypoint_tracking", {}).get(
            "commanded_index"
        )
        == 3
    ]
    assert len(delayed) == 1
    assert delayed[0]["waypoint_settled"] is True
    assert delayed[0]["orientation_error_rad"] == pytest.approx(
        math.radians(0.6), abs=1e-6
    )
    assert delayed[0]["violations"]["waypoint_orientation"] is False


def test_whole_body_subthreshold_lag_does_not_repeat_or_replan():
    class Backend(_FakeBackend):
        marker_step = 0.0001
        delayed_waypoint = 3

        def __init__(self):
            super().__init__()
            self.commanded_waypoint_indices = []

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(len(trajectory), dtype=np.float32) * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            return self._refresh_whole_body_certificate(result)

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            target = np.asarray(target_q, dtype=np.float32).reshape(-1)
            waypoint_index = int(round(float(target[6]) / self.marker_step))
            self.commanded_waypoint_indices.append(waypoint_index)
            nominal = self.execution_eef_positions[waypoint_index].astype(
                np.float64
            )
            if waypoint_index == self.delayed_waypoint:
                nominal = nominal - np.asarray([0.0, 0.0044, 0.0])
            self.next_execution_eef_pose = nominal
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                waypoint_index
            ].astype(np.float64)
            return action

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert backend.commanded_waypoint_indices[:5] == [0, 1, 2, 3, 4]
    assert len(backend.whole_body_plan_calls) == 1
    assert len(env.calls) == 20
    repeated = [
        row["whole_body_eef_waypoint_tracking"]
        for row in result["diagnostics"]["trace"]
        if row.get("whole_body_eef_waypoint_tracking", {}).get(
            "commanded_index"
        )
        == 3
    ]
    assert len(repeated) == 1
    assert repeated[0]["settle_repeat"] is False
    assert repeated[0]["prospective_position_bound_m"] > 0.005


def test_whole_body_raw_success_preempts_all_post_step_checks():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.feedback_reads = {
                "pose": 0,
                "attachment": 0,
                "whole_contact": 0,
                "dynamics": 0,
            }
            self.feedback_reads_at_success = None

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            nominal = self.execution_eef_positions[0].astype(np.float64)
            if self.env is not None and len(self.env.calls) == 0:
                nominal = nominal - np.asarray([0.0, 0.0044, 0.0])
            self.next_execution_eef_pose = nominal
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                0
            ].astype(np.float64)
            return action

        def get_eef_pose(self, hand):
            self.feedback_reads["pose"] += 1
            return super().get_eef_pose(hand)

        def get_attached_object(self, hand):
            self.feedback_reads["attachment"] += 1
            return super().get_attached_object(hand)

        def whole_body_contact_report(
            self,
            *,
            baseline,
            expected_attachments_by_hand,
            allowed_expected_contact=None,
        ):
            self.feedback_reads["whole_contact"] += 1
            return super().whole_body_contact_report(
                baseline=baseline,
                expected_attachments_by_hand=expected_attachments_by_hand,
                allowed_expected_contact=allowed_expected_contact,
            )

        def dynamics_report(self):
            raise AssertionError("whole-body runtime must not read dynamics")

    backend = Backend()
    executor, env = _executor(backend)

    def chunk_step(actions):
        env.calls.append(np.asarray(actions).copy())
        backend.advance()
        success = len(env.calls) == 2
        if success:
            backend.feedback_reads_at_success = dict(backend.feedback_reads)
        return (
            None,
            0.0,
            False,
            False,
            {
                "done": {"success": success},
                "_rpent": {"executed_steps": 1},
            },
        )

    env.chunk_step = chunk_step
    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["task_success"] is True
    assert len(env.calls) == 2
    terminal = result["diagnostics"]["trace"][-1]
    assert terminal["whole_body_waypoint_command"]["settle_repeat"] is False
    assert backend.feedback_reads == backend.feedback_reads_at_success
    assert backend.feedback_reads["dynamics"] == 0


def test_whole_body_next_waypoint_still_checks_unexpected_contact():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.whole_contact_calls = 0

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            nominal = self.execution_eef_positions[0].astype(np.float64)
            nominal = nominal - np.asarray([0.0, 0.0044, 0.0])
            self.next_execution_eef_pose = nominal
            self.next_execution_eef_quat = self.execution_eef_quaternions[
                0
            ].astype(np.float64)
            return action

        def whole_body_contact_report(
            self,
            *,
            baseline,
            expected_attachments_by_hand,
            allowed_expected_contact=None,
        ):
            del baseline, expected_attachments_by_hand, allowed_expected_contact
            self.whole_contact_calls += 1
            return {
                "available": True,
                "unexpected_contact": self.whole_contact_calls == 2,
                "unexpected_pairs": (
                    [["/World/robot/base", "/World/cabinet"]]
                    if self.whole_contact_calls == 2
                    else []
                ),
            }

    backend = Backend()
    executor, env = _executor(backend)
    result = executor.move_to(
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "unexpected_contact"
    assert len(env.calls) == 2
    command = result["diagnostics"]["trace"][-1][
        "whole_body_waypoint_command"
    ]
    assert command["settle_repeat"] is False
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_expected_contact_stops_on_current_certified_waypoint():
    backend = _FakeBackend(contact_mode="expected")
    executor, env = _executor(backend)
    target = np.asarray([0.53, 0.0, 0.0])
    plan = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        timeout_s=1.0,
        attached_obj=None,
    )

    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=math.radians(1.0),
        timeout_s=2.0,
        require_pose=True,
        hold_steps_required=5,
        contact_target_xyz=target,
        stop_on_expected_contact=True,
        joint_trajectory=plan["joint_trajectory"],
        expected_attachments_by_hand=plan["expected_attachments_by_hand"],
        motion_scope="whole_body",
        whole_body_certificate=plan["whole_body_certificate"],
        allow_replan_checkpoint=True,
    )

    assert result["primitive_success"] is True
    assert "whole_body_replan_checkpoint" not in result["metrics"]
    assert result["metrics"]["expected_contact_seen"] is True
    assert len(env.calls) == 1
    assert result["trace"][-1]["expected_contact_stop"] is True
    assert result["trace"][-1]["whole_body_waypoint_command"]["index"] == 0


def test_expected_contact_cannot_hide_other_whole_body_collision():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__(contact_mode="expected")
            self.allowed_expected_contacts = []

        def whole_body_contact_report(
            self,
            *,
            baseline,
            expected_attachments_by_hand,
            allowed_expected_contact=None,
        ):
            del baseline, expected_attachments_by_hand
            self.allowed_expected_contacts.append(allowed_expected_contact)
            return {
                "available": True,
                "unexpected_contact": True,
                "unexpected_pairs": [
                    ["/World/robot/base", "/World/cabinet"],
                ],
            }

    backend = Backend()
    executor, env = _executor(backend)
    target = np.asarray([0.53, 0.0, 0.0])
    plan = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        timeout_s=1.0,
        attached_obj=None,
    )

    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=math.radians(1.0),
        timeout_s=2.0,
        require_pose=True,
        hold_steps_required=5,
        contact_target_xyz=target,
        stop_on_expected_contact=True,
        joint_trajectory=plan["joint_trajectory"],
        expected_attachments_by_hand=plan["expected_attachments_by_hand"],
        motion_scope="whole_body",
        whole_body_certificate=plan["whole_body_certificate"],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "unexpected_contact"
    assert len(env.calls) == 1
    assert backend.allowed_expected_contacts[0]["expected_contact"] is True
    command = result["trace"][-1]["whole_body_waypoint_command"]
    assert command["index"] == 0
    assert command["q_sha256"]
    assert command["trajectory_sha256"]


@pytest.mark.parametrize(
    "contact_mode",
    ["expected_contact", "attachment_contact"],
)
def test_post_action_contact_unavailable_trace_binds_certified_waypoint(
    contact_mode,
):
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.attached_obj = {"left_eef_link": self.target_root}

        @staticmethod
        def capture_whole_body_contact_baseline(
            *,
            expected_attachments_by_hand,
        ):
            del expected_attachments_by_hand
            return {"available": True, "pairs": [], "continuous_pairs": []}

        @staticmethod
        def contact_report(
            *,
            hand,
            target_xyz=None,
            allowed_contact_distance_m=0.025,
        ):
            del hand, target_xyz, allowed_contact_distance_m
            return {
                "available": False,
                "reason": "injected post-action contact failure",
                "unexpected_contact": False,
                "expected_contact": False,
            }

    backend = Backend()
    executor, env = _executor(backend)
    env._gripper_latch["left"] = 0.0
    target = np.asarray([0.53, 0.0, 0.0])
    selected_attachment = backend.get_attached_object("left")
    plan = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        timeout_s=1.0,
        attached_obj=selected_attachment,
    )
    contact_kwargs = (
        {"stop_on_expected_contact": True}
        if contact_mode == "expected_contact"
        else {
            "stop_on_attachment": True,
            "expected_attachment": plan["expected_attachments_by_hand"]["left"],
        }
    )

    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=target,
        target_quat_xyzw=None,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=math.radians(1.0),
        timeout_s=2.0,
        require_pose=True,
        hold_steps_required=5,
        contact_target_xyz=target,
        joint_trajectory=plan["joint_trajectory"],
        expected_attachments_by_hand=plan["expected_attachments_by_hand"],
        motion_scope="whole_body",
        whole_body_certificate=plan["whole_body_certificate"],
        **contact_kwargs,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "contact_feedback_unavailable"
    assert len(env.calls) == 1
    command = result["trace"][-1]["whole_body_waypoint_command"]
    assert command["index"] == 0
    expected_q_sha256 = hashlib.sha256(
        np.ascontiguousarray(
            plan["joint_trajectory"][0],
            dtype=np.float32,
        ).tobytes()
    ).hexdigest()
    assert command["q_sha256"] == expected_q_sha256
    assert (
        command["trajectory_sha256"]
        == plan["whole_body_certificate"]["trajectory_sha256"]
    )
    assert result["metrics"]["post_stop_env_actions"] == 0


def test_whole_body_contact_report_exempts_only_proven_target_finger_pair():
    backend = RealCuroboBackend(None)
    expected_pair = tuple(
        sorted(("/World/robot/left_finger", "/World/button/collision"))
    )
    unrelated_pair = tuple(
        sorted(("/World/robot/base", "/World/cabinet/collision"))
    )
    baseline = {
        "available": True,
        "pairs": [],
        "continuous_pairs": [],
    }
    backend._whole_body_contact_pairs = lambda _attachments: {
        expected_pair,
        unrelated_pair,
    }

    report = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand={"left": None, "right": None},
        allowed_expected_contact={
            "expected_contact": True,
            "target_root": "/World/button",
            "target_finger_paths": ["/World/robot/left_finger"],
        },
    )

    assert report["available"] is True
    assert report["unexpected_contact"] is True
    assert report["allowed_expected_contact_pairs"] == [list(expected_pair)]
    assert report["unexpected_pairs"] == [list(unrelated_pair)]

    backend._whole_body_contact_pairs = lambda _attachments: {expected_pair}
    allowed_only = backend.whole_body_contact_report(
        baseline={
            "available": True,
            "pairs": [],
            "continuous_pairs": [],
        },
        expected_attachments_by_hand={"left": None, "right": None},
        allowed_expected_contact={
            "expected_contact": True,
            "target_root": "/World/button",
            "target_finger_paths": ["/World/robot/left_finger"],
        },
    )
    assert allowed_only["unexpected_contact"] is False


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_whole_body_nonfinite_live_eef_feedback_stops_after_current_action(
    invalid_value,
):
    class Backend(_FakeBackend):
        def advance(self):
            self.pose = np.asarray(
                [invalid_value, 0.0, 0.0],
                dtype=np.float64,
            )

    backend = Backend()
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="right",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "pose_feedback_unavailable"
    assert result["metrics"]["executed_waypoints"] == 1
    assert result["metrics"]["partial_motion"] is True
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert len(env.calls) == 1


def test_whole_body_live_wrist_backtracking_stops_before_next_action():
    class Backend(_FakeBackend):
        def advance(self):
            super().advance()
            assert self.env is not None
            executed = len(self.env.calls)
            if executed in {1, 2}:
                angle = math.radians(1.0 if executed == 1 else 0.0)
                self.quat = np.asarray(
                    [0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5)],
                    dtype=np.float64,
                )

    backend = Backend()
    executor, env = _executor(backend)
    target_angle = math.radians(5.0)

    result = executor.move_to(
        hand="right",
        target_xyz=backend.pose.copy(),
        target_quat_xyzw=[
            0.0,
            0.0,
            math.sin(target_angle * 0.5),
            math.cos(target_angle * 0.5),
        ],
        orientation_tolerance_rad=math.radians(1.0),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "eef_path_divergence"
    assert result["metrics"]["executed_waypoints"] == 2
    assert result["metrics"]["partial_motion"] is True
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert len(env.calls) == 2
    guard = result["metrics"]["whole_body_eef_path_guard"]
    assert guard["last_waypoint"]["violations"]["orientation_monotonic"] is True
    assert (
        guard["max_observed_live_orientation_reverse_progress_rad"]
        > math.radians(0.25)
    )


def test_whole_body_eef_path_admission_failure_executes_zero_actions():
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            self.whole_body_plan_calls.append(dict(kwargs))
            return {
                "ok": False,
                "stop_reason": "eef_path_admission_failed",
                "metrics": {
                    "candidate_audit": [
                        {
                            "certified": False,
                            "rejection_reason": "selected_eef_path_rejected",
                            "selected_eef_path": {
                                "available": True,
                                "admitted": False,
                                "max_start_excursion_m": 0.7,
                                "max_start_excursion_limit_m": 0.035,
                            },
                        }
                    ]
                },
            }

    executor, env = _executor(Backend())

    result = executor.move_to(
        hand="left",
        target_xyz=[0.47, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "replan_no_progress"
    assert result["metrics"]["last_plan_stop_reason"] == (
        "eef_path_admission_failed"
    )
    assert len(result["metrics"]["replan_rounds"]) == 2
    assert len(result["diagnostics"]["metrics"]["candidate_audit"]) == 1
    assert result["metrics"]["candidate_audit"][0]["selected_eef_path"][
        "max_start_excursion_m"
    ] == pytest.approx(0.7)
    assert len(executor.backend.whole_body_plan_calls) == 2
    assert env.calls == []


def test_whole_body_first_eef_admission_failure_replans_then_executes():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.plan_attempt = 0

        def plan_whole_body_trajectory(self, **kwargs):
            self.plan_attempt += 1
            if self.plan_attempt == 1:
                self.whole_body_plan_calls.append(dict(kwargs))
                return {
                    "ok": False,
                    "stop_reason": "eef_path_admission_failed",
                    "metrics": {
                        "env_actions_sent": 0,
                        "collision_admission": {
                            "available": False,
                            "admitted": False,
                        },
                    },
                }
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.47, 0.0, 0.0],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert result["primitive_success"] is True
    assert len(backend.whole_body_plan_calls) == 2
    assert len(result["metrics"]["replan_rounds"]) == 2
    assert result["metrics"]["replan_rounds"][0][
        "eligible_plan_failure"
    ]["unconditional_replan"] is True
    assert env.calls


def test_move_to_planning_retry_does_not_authorize_execution_failure_replan():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.plan_attempt = 0

        def plan_whole_body_trajectory(self, **kwargs):
            self.plan_attempt += 1
            if self.plan_attempt == 1:
                self.whole_body_plan_calls.append(dict(kwargs))
                return {
                    "ok": False,
                    "stop_reason": "eef_path_admission_failed",
                    "metrics": {"env_actions_sent": 0},
                }
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend()
    executor, env = _executor(backend)
    executor._execute_actions = lambda *_args, **_kwargs: {
        "primitive_success": False,
        "task_success": False,
        "stop_reason": "stalled_tracking",
        "recoverable": True,
        "metrics": {
            "env_actions_sent": 1,
            "executed_waypoints": 1,
            "partial_motion": True,
            "final_position_error_m": 0.02,
            "final_joint_tracking": {
                "normalized_21d_tracking_error": 1.0,
            },
        },
        "diagnostics": {"trace": []},
    }

    result = executor.move_to(
        hand="left",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["stop_reason"] == "stalled_tracking"
    assert len(backend.whole_body_plan_calls) == 2
    assert result["metrics"]["env_actions_sent"] == 1
    assert result["metrics"]["partial_motion"] is True
    assert env.calls == []


def test_move_to_checkpoint_then_planning_retry_preserves_action_accounting():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.plan_attempt = 0

        def plan_whole_body_trajectory(self, **kwargs):
            self.plan_attempt += 1
            if self.plan_attempt in {2, 3}:
                self.whole_body_plan_calls.append(dict(kwargs))
                return {
                    "ok": False,
                    "stop_reason": "eef_path_admission_failed",
                    "metrics": {"env_actions_sent": 0},
                }
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend()
    executor, env = _executor(backend)
    execute_calls = 0

    def execute(*_args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        return _scripted_whole_body_replan_checkpoint(
            kwargs,
            final_position_error_m=0.02,
        )

    executor._execute_actions = execute

    result = executor.move_to(
        hand="left",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["stop_reason"] == "replan_no_progress"
    assert execute_calls == 1
    assert len(backend.whole_body_plan_calls) == 3
    assert result["metrics"]["env_actions_sent"] == 1
    assert result["metrics"]["partial_motion"] is True
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert env.calls == []


@pytest.mark.parametrize(
    "certificate_fault",
    [
        "collision_admission",
        "missing_certificate",
        "trajectory_digest",
        "eef_target_hash",
        "eef_positions_digest",
        "eef_short_target_flag",
        "dense_trajectory_digest",
        "dense_waypoint_count",
        "joint_layout_digest",
        "execution_base_xy_step_limit",
        "execution_base_yaw_step_limit",
        "execution_articulation_step_limit",
        "terminal_command_limit",
        "terminal_position_tolerance",
        "terminal_orientation_tolerance",
    ],
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
            elif certificate_fault == "trajectory_digest":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "trajectory_sha256": "tampered",
                }
            elif certificate_fault == "eef_target_hash":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "selected_target_xyz_sha256": "0" * 64,
                }
            elif certificate_fault == "eef_positions_digest":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "selected_eef_execution_positions_sha256": "tampered",
                }
            elif certificate_fault == "eef_short_target_flag":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "selected_eef_short_target": False,
                }
            elif certificate_fault == "dense_trajectory_digest":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "dense_collision_trajectory_sha256": "tampered",
                }
            elif certificate_fault == "dense_waypoint_count":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "collision_free_waypoints": 999,
                }
            elif certificate_fault.startswith("execution_"):
                key = {
                    "execution_base_xy_step_limit": (
                        "execution_base_xy_step_limit_m"
                    ),
                    "execution_base_yaw_step_limit": (
                        "execution_base_yaw_step_limit_rad"
                    ),
                    "execution_articulation_step_limit": (
                        "execution_articulation_step_limit_rad"
                    ),
                }[certificate_fault]
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    key: -1.0,
                }
            elif certificate_fault.startswith("terminal_"):
                key = {
                    "terminal_command_limit": "terminal_command_limit",
                    "terminal_position_tolerance": (
                        "terminal_eef_position_tolerance_m"
                    ),
                    "terminal_orientation_tolerance": (
                        "terminal_eef_orientation_tolerance_rad"
                    ),
                }[certificate_fault]
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    key: -1,
                }
            elif certificate_fault.startswith("stationary_"):
                key = {
                    "stationary_base_translation_threshold": (
                        "waypoint_stationary_base_translation_"
                        "max_actual_velocity_m_s"
                    ),
                    "stationary_base_yaw_threshold": (
                        "waypoint_stationary_base_yaw_"
                        "max_actual_velocity_rad_s"
                    ),
                    "stationary_articulation_threshold": (
                        "waypoint_stationary_articulation_"
                        "max_actual_velocity_rad_s"
                    ),
                    "stationary_articulation_step": (
                        "waypoint_stationary_articulation_max_step_rad"
                    ),
                    "stationary_sample_rate": (
                        "waypoint_stationary_sample_rate_hz"
                    ),
                    "stationary_policy": "waypoint_stationary_policy",
                }[certificate_fault]
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    key: (
                        "tampered"
                        if certificate_fault == "stationary_policy"
                        else -1.0
                    ),
                }
            else:
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "joint_name_layout_sha256": "tampered",
                }
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(
        hand="left",
        target_xyz=[0.47, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == (
        "error"
        if certificate_fault
        not in {"collision_admission", "missing_certificate"}
        else "collision_admission_unavailable"
    )
    assert env.calls == []


@pytest.mark.parametrize(
    "certificate_fault",
    ["dense_trajectory_digest", "dense_waypoint_count"],
)
def test_long_move_to_binds_exact_execution_to_dense_collision_lineage(
    certificate_fault,
):
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            if certificate_fault == "dense_trajectory_digest":
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "dense_collision_trajectory_sha256": "tampered",
                }
            else:
                result["whole_body_certificate"] = {
                    **result["whole_body_certificate"],
                    "collision_free_waypoints": 999,
                }
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(
        hand="left",
        target_xyz=[0.45, 0.0, -0.05],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "does not match its collision certificate" in result["diagnostics"]["error"]
    assert env.calls == []


def test_whole_body_ordered_execution_subset_sends_only_certified_rows():
    class Backend(_FakeBackend):
        marker_step = 0.005

        def __init__(self):
            super().__init__()
            self.last_plan = None
            self.runtime_q_targets = []

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(1, len(trajectory) + 1, dtype=np.float32)
                * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            self.last_plan = self._refresh_whole_body_certificate(result)
            return self.last_plan

        def joint_target_to_action(
            self,
            target_q,
            *,
            hand,
            fixed_reference=None,
        ):
            self.runtime_q_targets.append(
                np.asarray(target_q, dtype=np.float32).copy()
            )
            return super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )

    backend = Backend()
    executor, env = _executor(backend)

    result = _move_to_without_replan_checkpoint(
        executor,
        hand="left",
        target_xyz=[0.53, 0.0, 0.0],
    )

    assert result["primitive_success"] is True
    assert backend.last_plan is not None
    execution = np.asarray(
        backend.last_plan["joint_trajectory"], dtype=np.float32
    )
    certificate = backend.last_plan["whole_body_certificate"]
    source_dense = np.asarray(
        certificate["source_dense_trajectory"], dtype=np.float32
    )
    collision_dense = np.asarray(
        certificate["dense_collision_trajectory"], dtype=np.float32
    )
    source_indices = np.asarray(
        certificate["execution_source_dense_indices"], dtype=np.int64
    )
    collision_indices = np.asarray(
        certificate["execution_collision_dense_indices"], dtype=np.int64
    )
    assert len(source_dense) > len(execution) + 1
    assert len(collision_dense) >= len(execution) + 1
    assert certificate["dense_collision_checked_waypoint_count"] == len(
        collision_dense
    )
    assert certificate["world_collision_check"] is True
    assert certificate["self_collision_check"] is True
    np.testing.assert_array_equal(
        source_dense[source_indices[1:]],
        execution,
    )
    np.testing.assert_array_equal(
        collision_dense[collision_indices[1:]],
        execution,
    )
    np.testing.assert_array_equal(
        np.asarray(backend.runtime_q_targets),
        execution,
    )
    assert len(env.calls) == len(execution)


@pytest.mark.parametrize(
    "fault",
    [
        "source_indices",
        "collision_indices",
        "dense_payload",
        "dense_digest",
        "execution_row",
        "start_q",
    ],
)
def test_whole_body_subset_lineage_tamper_fails_before_first_action(fault):
    class Backend(_FakeBackend):
        marker_step = 0.005

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(1, len(trajectory) + 1, dtype=np.float32)
                * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            result = self._refresh_whole_body_certificate(result)
            certificate = result["whole_body_certificate"]

            def indices_digest(values):
                return hashlib.sha256(
                    np.ascontiguousarray(values, dtype="<i8").tobytes()
                ).hexdigest()

            if fault == "source_indices":
                indices = np.asarray(
                    certificate["execution_source_dense_indices"],
                    dtype=np.int64,
                )
                indices[1] += 1
                certificate["execution_source_dense_indices"] = indices.tolist()
                certificate["execution_source_dense_indices_sha256"] = (
                    indices_digest(indices)
                )
            elif fault == "collision_indices":
                indices = np.asarray(
                    certificate["execution_collision_dense_indices"],
                    dtype=np.int64,
                )
                indices[1] -= 1
                certificate["execution_collision_dense_indices"] = (
                    indices.tolist()
                )
                certificate["execution_collision_dense_indices_sha256"] = (
                    indices_digest(indices)
                )
            elif fault == "dense_payload":
                dense = np.asarray(
                    certificate["dense_collision_trajectory"],
                    dtype=np.float32,
                )
                dense[1, 6] += np.float32(0.0001)
                certificate["dense_collision_trajectory"] = dense.tolist()
            elif fault == "dense_digest":
                certificate["dense_collision_trajectory_sha256"] = "tampered"
            elif fault == "execution_row":
                execution = np.asarray(
                    result["joint_trajectory"], dtype=np.float32
                ).copy()
                execution[0, 6] += np.float32(0.0001)
                result["joint_trajectory"] = execution
                execution_digest = hashlib.sha256(
                    np.ascontiguousarray(execution, dtype=np.float32).tobytes()
                ).hexdigest()
                certificate["trajectory_sha256"] = execution_digest
                certificate["execution_trajectory_sha256"] = execution_digest
            else:
                self.joint_positions[6] += np.float32(0.0001)
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(hand="left", target_xyz=[0.53, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "does not match its collision certificate" in (
        result["diagnostics"]["error"]
    )
    assert env.calls == []


def test_whole_body_full_dense_eef_payload_is_verified_before_first_action():
    class Backend(_FakeBackend):
        marker_step = 0.01

        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            trajectory = np.asarray(
                result["joint_trajectory"], dtype=np.float32
            ).copy()
            trajectory[:, 6] = (
                np.arange(1, len(trajectory) + 1, dtype=np.float32)
                * self.marker_step
            )
            result["joint_trajectory"] = trajectory
            result = self._refresh_whole_body_certificate(result)
            certificate = result["whole_body_certificate"]
            dense_positions = np.asarray(
                certificate["selected_eef_dense_positions"],
                dtype=np.float32,
            )
            collision_indices = set(
                certificate["execution_collision_dense_indices"]
            )
            tamper_index = next(
                index
                for index in range(1, len(dense_positions) - 1)
                if index not in collision_indices
            )
            dense_positions[tamper_index, 1] += np.float32(0.0001)
            digest = hashlib.sha256(
                np.ascontiguousarray(
                    dense_positions, dtype=np.float32
                ).tobytes()
            ).hexdigest()
            certificate["selected_eef_dense_positions"] = (
                dense_positions.tolist()
            )
            certificate["selected_eef_dense_positions_sha256"] = digest
            certificate["selected_eef_positions_sha256"] = digest
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(hand="left", target_xyz=[0.53, 0.0, 0.0])

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "recomputed_dense_position_digest" in result["diagnostics"]["error"]
    assert env.calls == []


def test_move_to_rejects_runtime_joint_layout_reordering_before_first_action():
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(self, **kwargs):
            result = super().plan_whole_body_trajectory(**kwargs)
            self.joint_names = tuple(reversed(self.joint_names))
            return result

    executor, env = _executor(Backend())

    result = executor.move_to(
        hand="right",
        target_xyz=[0.47, 0.0, 0.0],
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "collision certificate" in result["diagnostics"]["error"]
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
        result = (
            floor_map.clone()
            if hasattr(floor_map, "clone")
            else np.asarray(floor_map).copy()
        )
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
        yaw_steps = np.asarray(
            [
                _wrap_angle(float(delta))
                for delta in np.diff(
                    np.concatenate([[q_reference[5]], q_path[:, 5]])
                )
            ]
        )
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
        self.navigation_targets: list[np.ndarray] = []
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
        self.base_goal = np.array([0.3, 0.2, 0.2], dtype=np.float32)

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
            "base_goal_xyyaw_sha256": hashlib.sha256(
                np.ascontiguousarray(self.base_goal).tobytes()
            ).hexdigest(),
            "terminal_q_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    self.q_path[-1], dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "terminal_command_limit": TERMINAL_COMMAND_LIMIT,
            "terminal_position_tolerance_m": (
                BASE_TERMINAL_POSITION_TOLERANCE_M
            ),
            "terminal_orientation_tolerance_rad": (
                BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD
            ),
        }
        return {
            "ok": True,
            "joint_trajectory": self.q_path.copy(),
            "base_goal": self.base_goal.tolist(),
            "expected_attachments_by_hand": {"left": None, "right": None},
            "metrics": {
                "obstacle_refresh": {
                    "mode": "pose_only",
                    "count": 7,
                    "elapsed_s": 0.002,
                    "fallback": False,
                },
                "solver_stages": [
                    {
                        "name": "base_ik",
                        "elapsed_s": 0.01,
                        "timing_s": {"certificate_s": 0.009},
                    }
                ],
                "selected_solver_stage": "base_ik",
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
        plan = self._navigation_plan()
        attempt_cap = kwargs.get("base_attempt_timeout_cap_s")
        solver_cap = kwargs.get("base_solver_timeout_cap_s")
        if attempt_cap is not None and solver_cap is not None:
            hard_attempt = min(float(kwargs["timeout_s"]), float(attempt_cap))
            plan["metrics"].update(
                {
                    "attempt_timeout_cap_s": float(attempt_cap),
                    "solver_timeout_cap_s": float(solver_cap),
                    "attempt_timeout_budget_s": hard_attempt,
                    "attempt_timeout_s": min(
                        hard_attempt,
                        float(solver_cap),
                    ),
                    "solver_timeout_s": min(
                        hard_attempt,
                        float(solver_cap),
                    ),
                    "planning_profile": kwargs.get(
                        "base_planning_profile"
                    ),
                    "base_solver_deadline_enforcement": {
                        "solver_timeout_enforced": True,
                        "hard_wall_clock_enforced": not bool(
                            kwargs.get("background", False)
                        ),
                        "hard_wall_clock_deadline_s": (
                            None
                            if kwargs.get("background", False)
                            else hard_attempt
                        ),
                        "soft_deadline_s": (
                            hard_attempt
                            if kwargs.get("background", False)
                            else None
                        ),
                    },
                }
            )
        return plan

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
        self.navigation_targets.append(target_q.copy())
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
            "max_articulation_error_rad": 0.0,
        }


class _NavigationDynamicsBackend(_NavigationBackend):
    def __init__(self, dynamics_reports, isolation_reports=None):
        super().__init__(isolation_reports=isolation_reports)
        self.dynamics_reports = list(dynamics_reports)
        self.dynamics_calls = 0
        self.advance_calls = 0

    def advance(self):
        super().advance()
        self.advance_calls += 1

    def dynamics_report(self):
        self.dynamics_calls += 1
        assert self.dynamics_reports, "unexpected navigation dynamics read"
        return self.dynamics_reports.pop(0)


class _NavigationSettleGateBackend(_NavigationDynamicsBackend):
    def __init__(self, dynamics_reports, *, settle_fault):
        super().__init__(dynamics_reports)
        self.settle_fault = settle_fault
        self.settle_events: list[str] = []

    def _is_first_terminal_settle_hold(self):
        return self.advance_calls == len(self.q_path) + 1

    def dynamics_report(self):
        if self._is_first_terminal_settle_hold():
            self.settle_events.append("dynamics")
        return super().dynamics_report()

    def joint_tracking_report(self, target_q, *, hand):
        if self._is_first_terminal_settle_hold():
            self.settle_events.append("tracking")
            if self.settle_fault == "tracking_not_reached":
                return {
                    "available": True,
                    "reached": False,
                    "max_base_xy_error_m": 0.02,
                    "base_yaw_error_rad": 0.0,
                }
        return super().joint_tracking_report(target_q, hand=hand)

    def get_base_pose(self):
        if self._is_first_terminal_settle_hold():
            self.settle_events.append("pose")
            if self.settle_fault == "pose_outside_tolerance":
                return self.base_goal.astype(np.float64) + [0.02, 0.0, 0.0]
        return super().get_base_pose()


def _navigation_isolation_sample(
    *,
    roll_pitch_drift_rad=0.0,
    trunk_drift_rad=0.0,
    failed_checks=(),
):
    checks = {
        "base_z_locked": True,
        "base_roll_pitch_locked": True,
        "trunk_locked": True,
        "left_arm_locked": True,
        "right_arm_locked": True,
        "left_gripper_command_locked": True,
        "right_gripper_command_locked": True,
        "left_attachment_identity_unchanged": True,
        "right_attachment_identity_unchanged": True,
    }
    for name in failed_checks:
        checks[name] = False
    return {
        "available": True,
        "ok": all(checks.values()),
        "mode": "base_only",
        "checks": checks,
        "max_observed": {
            "base_roll_pitch_drift_rad": float(roll_pitch_drift_rad),
            "trunk_drift_rad": float(trunk_drift_rad),
        },
        "thresholds": {
            "base_roll_pitch_rad": math.radians(1.0),
            "articulation_rad": 0.01,
        },
    }


@pytest.mark.parametrize(
    ("initial_drift", "settled_drift"),
    [
        (math.radians(1.0) + 2e-9, math.radians(0.75)),
        (math.radians(1.5), math.radians(1.0)),
    ],
)
def test_dashboard_base_terminal_roll_pitch_transient_settles_once(
    initial_drift,
    settled_drift,
):
    reports = [
        _navigation_isolation_sample(),
        _navigation_isolation_sample(),
        _navigation_isolation_sample(
            roll_pitch_drift_rad=initial_drift,
            failed_checks=("base_roll_pitch_locked",),
        ),
        _navigation_isolation_sample(
            roll_pitch_drift_rad=settled_drift,
        ),
    ]
    backend = _NavigationBackend(reports)
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "backward")
    assert prepared["execution_policy"] == (
        PREPARED_DASHBOARD_BASE_EXECUTION_POLICY
    )

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-terminal-tilt",
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.calls) == len(backend.q_path) + 1
    np.testing.assert_array_equal(
        backend.navigation_targets[-2],
        backend.navigation_targets[-1],
    )
    settle = result["metrics"]["dashboard_base_terminal_tilt_settle"]
    assert settle["eligible"] is True
    assert settle["hold_sent"] is True
    assert settle["maximum_holds"] == 1
    assert settle["initial_base_roll_pitch_drift_rad"] == pytest.approx(
        initial_drift
    )
    assert settle["settled_base_roll_pitch_drift_rad"] == pytest.approx(
        settled_drift
    )
    assert settle["settled"] is True


def test_dashboard_base_terminal_roll_pitch_must_settle_within_one_hold():
    drift = math.radians(1.25)
    reports = [
        _navigation_isolation_sample(),
        _navigation_isolation_sample(),
        _navigation_isolation_sample(
            roll_pitch_drift_rad=drift,
            failed_checks=("base_roll_pitch_locked",),
        ),
        _navigation_isolation_sample(
            roll_pitch_drift_rad=drift,
            failed_checks=("base_roll_pitch_locked",),
        ),
    ]
    backend = _NavigationBackend(reports)
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "backward")

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-terminal-tilt-not-settled",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_violation"
    assert len(env.calls) == len(backend.q_path) + 1
    assert len(backend.isolation_reports) == 0
    assert result["metrics"]["post_stop_env_actions"] == 0


@pytest.mark.parametrize(
    "drift",
    [
        math.radians(1.0) + 2e-9,
        math.radians(1.5),
    ],
)
def test_dashboard_base_intermediate_roll_pitch_transient_continues(
    drift,
):
    backend = _NavigationBackend(
        [
            _navigation_isolation_sample(
                roll_pitch_drift_rad=drift,
                failed_checks=("base_roll_pitch_locked",),
            ),
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
        ]
    )
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "backward")

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-intermediate-tilt",
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.calls) == len(backend.q_path)
    transient = result["metrics"]["dashboard_base_terminal_tilt_settle"]
    assert transient["intermediate_transient_count"] == 1
    assert transient[
        "maximum_intermediate_base_roll_pitch_drift_rad"
    ] == pytest.approx(drift)
    assert result["trace"][0]["navigation_isolation_deferred"] == (
        "dashboard_base_intermediate_roll_pitch_transient"
    )


@pytest.mark.parametrize("drift", [0.010000002, 0.015])
def test_dashboard_base_intermediate_trunk_transient_continues(drift):
    backend = _NavigationBackend(
        [
            _navigation_isolation_sample(
                trunk_drift_rad=drift,
                failed_checks=("trunk_locked",),
            ),
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
        ]
    )
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "forward")

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-intermediate-trunk",
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.calls) == len(backend.q_path)
    transient = result["metrics"]["dashboard_base_terminal_tilt_settle"]
    assert transient["normal_articulation_limit_rad"] == pytest.approx(0.01)
    assert transient["transient_trunk_limit_rad"] == pytest.approx(0.015)
    assert transient["intermediate_trunk_transient_count"] == 1
    assert transient["maximum_intermediate_trunk_drift_rad"] == pytest.approx(
        drift
    )
    isolation = result["metrics"]["navigation_isolation"]
    assert isolation["ok"] is True
    assert isolation["all_steps_strict_ok"] is False
    assert isolation["failed_checks_observed"] == ["trunk_locked"]
    assert isolation["checks_performed"] == len(backend.q_path)
    assert isolation["max_observed"]["trunk_drift_rad"] == pytest.approx(drift)
    assert result["trace"][0]["navigation_isolation_deferred"] == (
        "dashboard_base_intermediate_trunk_transient"
    )


@pytest.mark.parametrize(
    ("drift", "prepared"),
    [
        (0.015000002, True),
        (0.012, False),
    ],
)
def test_dashboard_base_intermediate_trunk_transient_is_narrow(
    drift,
    prepared,
):
    backend = _NavigationBackend(
        [
            _navigation_isolation_sample(
                trunk_drift_rad=drift,
                failed_checks=("trunk_locked",),
            ),
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
        ]
    )
    executor, env = _executor(backend)
    if prepared:
        plan = executor.prepare_dashboard_motion("chassis", "forward")
        result = executor.execute_dashboard_motion(
            plan["plan_id"],
            "base-intermediate-trunk-rejected",
        )
    else:
        result = executor.navigate_to(
            target_xyz=[1.0, 0.0, 0.0],
            standoff_m=0.85,
            max_travel_m=1.0,
            timeout_s=5.0,
        )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_violation"
    assert len(env.calls) == 1
    assert result["metrics"]["post_stop_env_actions"] == 0
    isolation = result["metrics"]["navigation_isolation"]
    assert isolation["available"] is True
    assert isolation["ok"] is False
    assert isolation["all_steps_strict_ok"] is False
    assert isolation["failed_checks_observed"] == ["trunk_locked"]
    assert isolation["checks_performed"] == 1
    assert isolation["max_observed"]["trunk_drift_rad"] == pytest.approx(drift)


def test_dashboard_base_terminal_trunk_drift_remains_strict():
    drift = 0.01131296157836914
    backend = _NavigationBackend(
        [
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
            _navigation_isolation_sample(
                trunk_drift_rad=drift,
                failed_checks=("trunk_locked",),
            ),
        ]
    )
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "forward")

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-terminal-trunk-rejected",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_violation"
    assert len(env.calls) == len(backend.q_path)
    settle = result["metrics"]["dashboard_base_terminal_tilt_settle"]
    assert settle["hold_sent"] is False
    isolation = result["metrics"]["navigation_isolation"]
    assert isolation["ok"] is False
    assert isolation["all_steps_strict_ok"] is False
    assert isolation["failed_checks_observed"] == ["trunk_locked"]
    assert isolation["checks_performed"] == len(backend.q_path)
    assert isolation["max_observed"]["trunk_drift_rad"] == pytest.approx(drift)
    assert result["metrics"]["post_stop_env_actions"] == 0


@pytest.mark.parametrize(
    ("report", "prepared"),
    [
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.5) + 2e-9,
                failed_checks=("base_roll_pitch_locked",),
            ),
            True,
        ),
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.25),
                failed_checks=("base_roll_pitch_locked", "trunk_locked"),
            ),
            True,
        ),
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.25),
                failed_checks=("base_roll_pitch_locked",),
            ),
            False,
        ),
    ],
)
def test_dashboard_base_intermediate_roll_pitch_transient_is_narrow(
    report,
    prepared,
):
    backend = _NavigationBackend(
        [
            report,
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
        ]
    )
    executor, env = _executor(backend)
    if prepared:
        plan = executor.prepare_dashboard_motion("chassis", "backward")
        result = executor.execute_dashboard_motion(
            plan["plan_id"],
            "base-intermediate-tilt-rejected",
        )
    else:
        result = executor.navigate_to(
            target_xyz=[1.0, 0.0, 0.0],
            standoff_m=0.85,
            max_travel_m=1.0,
            timeout_s=5.0,
        )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_violation"
    assert len(env.calls) == 1
    assert result["metrics"]["post_stop_env_actions"] == 0


@pytest.mark.parametrize(
    ("terminal_report", "prepared", "expected_actions"),
    [
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.5) + 2e-9,
                failed_checks=("base_roll_pitch_locked",),
            ),
            True,
            3,
        ),
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.25),
                failed_checks=(
                    "base_roll_pitch_locked",
                    "trunk_locked",
                ),
            ),
            True,
            3,
        ),
        (
            _navigation_isolation_sample(
                roll_pitch_drift_rad=math.radians(1.25),
                failed_checks=("base_roll_pitch_locked",),
            ),
            False,
            3,
        ),
    ],
)
def test_dashboard_base_terminal_roll_pitch_policy_does_not_broaden_other_paths(
    terminal_report,
    prepared,
    expected_actions,
):
    backend = _NavigationBackend(
        [
            _navigation_isolation_sample(),
            _navigation_isolation_sample(),
            terminal_report,
        ]
    )
    executor, env = _executor(backend)
    if prepared:
        plan = executor.prepare_dashboard_motion("chassis", "backward")
        result = executor.execute_dashboard_motion(
            plan["plan_id"],
            "base-terminal-tilt-rejected",
        )
    else:
        result = executor.navigate_to(
            target_xyz=[1.0, 0.0, 0.0],
            standoff_m=0.85,
            max_travel_m=1.0,
            timeout_s=5.0,
        )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_violation"
    assert len(env.calls) == expected_actions
    assert len(backend.isolation_reports) == 0
    assert result["metrics"]["post_stop_env_actions"] == 0


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


def test_prepared_dashboard_queue_is_picklable_predicted_and_exactly_once():
    backend = _NavigationBackend()
    executor, env = _executor(backend)

    first = executor.prepare_dashboard_motion(
        "chassis",
        "forward",
        background=True,
    )
    assert pickle.loads(pickle.dumps(first)) == first
    assert first["status"] == "prepared"
    assert first["planning_profile"] == (
        DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
    )
    assert first["planning_deadline_s"] == pytest.approx(12.0)
    assert first["fast_solver_deadline_s"] == pytest.approx(4.0)
    assert first["plan_metrics"] == {
        "obstacle_refresh": {
            "mode": "pose_only",
            "count": 7,
            "elapsed_s": 0.002,
            "fallback": False,
        },
        "solver_stages": [{"name": "base_ik", "elapsed_s": 0.01}],
        "selected_solver_stage": "base_ik",
    }
    assert first["deadline_enforcement"] == {
        "solver_timeout_enforced": True,
        "hard_wall_clock_enforced": False,
        "hard_wall_clock_deadline_s": None,
        "soft_deadline_s": 12.0,
        "soft_deadline_exceeded": False,
        "base_attempt_timeout_budget_s": 4.0,
        "base_solver_timeout_s": 4.0,
        "base_attempt_timeout_cap_s": 4.0,
        "base_solver_timeout_cap_s": 4.0,
        "base_solver_deadline_enforcement": {
            "solver_timeout_enforced": True,
            "hard_wall_clock_enforced": False,
            "hard_wall_clock_deadline_s": None,
            "soft_deadline_s": 4.0,
        },
    }
    assert "joint_trajectory" not in first

    second = executor.prepare_dashboard_motion(
        "chassis",
        "turn_left",
        predecessor_plan_id=first["plan_id"],
        background=True,
    )
    np.testing.assert_allclose(
        backend.relative_navigation_calls[1]["start_q"],
        first["predicted_terminal"]["joint_positions"],
    )
    np.testing.assert_allclose(
        backend.relative_navigation_calls[1]["start_base_xyyaw"],
        first["predicted_terminal"]["base_xyyaw"],
    )
    assert second["predecessor_plan_id"] == first["plan_id"]

    executed = executor.execute_dashboard_motion(first["plan_id"], "command-1")
    assert executed["primitive_success"] is True
    assert executed["metrics"]["prepared_plan_reused"] is True
    assert executed["metrics"]["live_start_checked"] is True
    assert executed["metrics"]["collision_revalidated"] is False
    assert executed["metrics"]["live_start_equality_skipped"] is False
    assert executed["metrics"]["prepared_start_strict_float32_equal"] is True
    assert executed["metrics"]["replan_required"] is False
    assert executed["metrics"]["collision_revalidation_skipped"] is True
    action_count = len(env.calls)

    replay = executor.execute_dashboard_motion(first["plan_id"], "command-1")
    assert replay == executed
    assert len(env.calls) == action_count
    with pytest.raises(RuntimeError, match="different command"):
        executor.execute_dashboard_motion(first["plan_id"], "command-2")
    consumed = executor.discard_dashboard_motion(first["plan_id"])
    assert consumed["discarded"] is False
    assert consumed["status"] == "completed"

    discarded = executor.discard_dashboard_motion(second["plan_id"])
    assert discarded["discarded"] is True
    with pytest.raises(RuntimeError, match="not executable"):
        executor.execute_dashboard_motion(second["plan_id"], "command-3")


def test_prepared_base_live_start_drift_requires_replan_without_actions():
    backend = _NavigationBackend()
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("chassis", "forward")
    backend.joint_positions[6] = np.nextafter(
        np.float32(0.0),
        np.float32(np.inf),
    )

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-start-drift",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "prepared_start_drift"
    assert result["recoverable"] is True
    assert result["metrics"]["replan_required"] is True
    assert result["metrics"]["live_start_checked"] is True
    assert result["metrics"]["live_start_equality_skipped"] is False
    assert result["metrics"]["prepared_start_strict_float32_equal"] is False
    assert result["metrics"]["env_actions_sent"] == 0
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert result["metrics"]["planned_start_q_sha256"] != (
        result["metrics"]["live_start_q_sha256"]
    )
    assert env.calls == []
    assert backend.navigation_targets == []
    assert executor._prepared_motions[prepared["plan_id"]]["status"] == "failed"

    replay = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "base-start-drift",
    )
    assert replay == result
    assert env.calls == []
    with pytest.raises(RuntimeError, match="different command"):
        executor.execute_dashboard_motion(
            prepared["plan_id"],
            "base-start-drift-other",
        )


def test_prepared_eef_live_start_drift_requires_replan_without_actions():
    class Backend(_FakeBackend):
        def plan_whole_body_trajectory(
            self,
            *,
            start_q=None,
            start_eef_pose=None,
            background=False,
            **kwargs,
        ):
            del start_q, start_eef_pose, background
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend()
    executor, env = _executor(backend)
    prepared = executor.prepare_dashboard_motion("left_arm", "forward")
    backend.joint_positions[6] = np.nextafter(
        np.float32(0.0),
        np.float32(np.inf),
    )

    result = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "eef-start-drift",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "prepared_start_drift"
    assert result["recoverable"] is True
    assert result["metrics"]["replan_required"] is True
    assert result["metrics"]["live_start_checked"] is True
    assert result["metrics"]["prepared_start_strict_float32_equal"] is False
    assert result["metrics"]["env_actions_sent"] == 0
    assert result["metrics"]["post_stop_env_actions"] == 0
    assert env.calls == []
    assert backend.whole_body_hold_calls == []


def test_successful_predecessor_actual_terminal_drift_rejects_successor():
    class Backend(_NavigationBackend):
        def plan_relative_navigation_trajectory(self, **kwargs):
            plan = super().plan_relative_navigation_trajectory(**kwargs)
            start_q = np.ascontiguousarray(
                np.asarray(kwargs["start_q"], dtype=np.float32)
            )
            plan["metrics"]["base_trajectory_certificate"][
                "start_q_sha256"
            ] = hashlib.sha256(start_q.tobytes()).hexdigest()
            return plan

    backend = Backend()
    executor, env = _executor(backend)
    first = executor.prepare_dashboard_motion("chassis", "forward")
    successor = executor.prepare_dashboard_motion(
        "chassis",
        "turn_left",
        predecessor_plan_id=first["plan_id"],
    )

    first_result = executor.execute_dashboard_motion(
        first["plan_id"],
        "predecessor",
    )
    assert first_result["primitive_success"] is True
    actions_after_predecessor = len(env.calls)
    targets_after_predecessor = len(backend.navigation_targets)
    assert actions_after_predecessor > 0
    assert not np.array_equal(
        backend.joint_positions,
        np.asarray(
            first["predicted_terminal"]["joint_positions"],
            dtype=np.float32,
        ),
    )

    result = executor.execute_dashboard_motion(
        successor["plan_id"],
        "successor",
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "prepared_start_drift"
    assert result["metrics"]["replan_required"] is True
    assert result["metrics"]["env_actions_sent"] == 0
    assert result["metrics"]["predecessor_plan_id"] == first["plan_id"]
    assert len(env.calls) == actions_after_predecessor
    assert len(backend.navigation_targets) == targets_after_predecessor


def test_prepared_base_foreground_wraps_relative_plan_and_certification_in_12s(
    monkeypatch,
):
    deadline_calls = []
    deadline_active = False

    @contextmanager
    def recording_deadline(timeout_s, operation):
        nonlocal deadline_active
        deadline_calls.append((str(operation), float(timeout_s)))
        deadline_active = True
        try:
            yield
        finally:
            deadline_active = False

    class Backend(_NavigationBackend):
        def plan_relative_navigation_trajectory(self, **kwargs):
            assert deadline_active is True
            return super().plan_relative_navigation_trajectory(**kwargs)

    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        recording_deadline,
    )
    backend = Backend()
    executor, _env = _executor(backend)

    prepared = executor.prepare_dashboard_motion(
        "chassis",
        "forward",
        background=False,
    )

    assert deadline_calls == [
        (
            "prepared Dashboard BASE planning transaction",
            pytest.approx(12.0),
        )
    ]
    call = backend.relative_navigation_calls[0]
    assert call["timeout_s"] == pytest.approx(12.0)
    assert call["base_attempt_timeout_cap_s"] == pytest.approx(4.0)
    assert call["base_solver_timeout_cap_s"] == pytest.approx(4.0)
    assert call["base_planning_profile"] == (
        DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
    )
    assert prepared["deadline_enforcement"]["hard_wall_clock_enforced"] is True
    assert prepared["deadline_enforcement"][
        "hard_wall_clock_deadline_s"
    ] == pytest.approx(12.0)
    assert prepared["deadline_enforcement"]["soft_deadline_s"] is None


def test_prepared_base_background_uses_soft_12s_without_signal_and_reports_overrun(
    monkeypatch,
):
    def forbidden_deadline(*_args, **_kwargs):
        raise AssertionError("background preparation must not install SIGALRM")

    ticks = iter((100.0, 112.000001))
    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        forbidden_deadline,
    )
    monkeypatch.setattr(
        "robots.behavior.planner_executor.time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    executor, _env = _executor(_NavigationBackend())

    prepared = executor.prepare_dashboard_motion(
        "chassis",
        "forward",
        background=True,
    )

    deadline = prepared["deadline_enforcement"]
    assert prepared["status"] == "prepared"
    assert deadline["hard_wall_clock_enforced"] is False
    assert deadline["hard_wall_clock_deadline_s"] is None
    assert deadline["soft_deadline_s"] == pytest.approx(12.0)
    assert deadline["soft_deadline_exceeded"] is True


def test_prepared_base_rejects_untruthful_4s_solver_metrics_without_caching():
    class Backend(_NavigationBackend):
        def plan_relative_navigation_trajectory(self, **kwargs):
            plan = super().plan_relative_navigation_trajectory(**kwargs)
            plan["metrics"]["solver_timeout_s"] = 4.000001
            return plan

    executor, env = _executor(Backend())

    with pytest.raises(RuntimeError, match="violated its 4s solver"):
        executor.prepare_dashboard_motion("chassis", "forward")

    assert executor._prepared_motions == {}
    assert env.calls == []


def test_prepared_turn_right_crossing_negative_pi_has_one_canonical_goal_hash():
    class CollisionGenerator:
        @staticmethod
        def check_collisions(q, **kwargs):
            assert kwargs["self_collision_check"] is True
            return np.zeros(len(q), dtype=bool)

    class Backend(_NavigationBackend):
        def __init__(self):
            super().__init__()
            self.joint_positions[5] = math.radians(-179.0)
            self.base_pose = np.asarray(
                [0.0, 0.0, self.joint_positions[5]],
                dtype=np.float64,
            )
            self.raw_goal = np.asarray(
                [0.0, 0.0, self.joint_positions[5] - math.radians(5.0)],
                dtype=np.float64,
            )
            with_start, _metrics = _minimum_jerk_base_execution_trajectory(
                np.stack(
                    [
                        self.joint_positions.astype(np.float64),
                        np.asarray(
                            [
                                *self.joint_positions[:5],
                                self.raw_goal[2],
                                *self.joint_positions[6:],
                            ],
                            dtype=np.float64,
                        ),
                    ]
                ),
                base_indices=list(range(6)),
            )
            self.q_path = with_start[1:]
            self.base_goal = _canonical_base_xyyaw(self.raw_goal)
            self._torch = SimpleNamespace(
                float32=np.float32,
                as_tensor=lambda value, dtype=None: np.asarray(
                    value,
                    dtype=np.float32 if dtype is not None else None,
                ),
            )
            self._collision_generator = CollisionGenerator()
            self._certifier_robot = SimpleNamespace(base_idx=np.arange(6))

        def _find_robot(self):
            return self._certifier_robot

        def _generator(self, **_kwargs):
            return self._collision_generator

        @staticmethod
        def _all_attached_objects(**_kwargs):
            return None, {"left": None, "right": None}

        def _navigation_plan(self):
            q_path, metrics, attachments = (
                RealCuroboBackend._certify_base_trajectory(
                    self,
                    self.q_path,
                    start_q=self.joint_positions,
                    base_goal_xyyaw=self.raw_goal,
                    skip_obstacle_update=False,
                )
            )
            return {
                "ok": True,
                "joint_trajectory": q_path,
                "base_goal": self.base_goal.copy(),
                "expected_attachments_by_hand": attachments,
                "metrics": metrics,
            }

    backend = Backend()
    executor, env = _executor(backend)

    prepared = executor.prepare_dashboard_motion(
        "chassis",
        "turn_right",
    )
    entry = executor._prepared_motions[prepared["plan_id"]]
    canonical_goal = _canonical_base_xyyaw(backend.raw_goal)
    certificate = entry["plan"]["metrics"]["base_trajectory_certificate"]

    assert backend.raw_goal[2] < -math.pi
    assert canonical_goal[2] > 0.0
    np.testing.assert_allclose(entry["plan"]["base_goal"], canonical_goal)
    np.testing.assert_allclose(
        prepared["predicted_terminal"]["base_xyyaw"],
        canonical_goal,
    )
    assert certificate["base_goal_xyyaw_sha256"] == (
        _whole_body_target_sha256(canonical_goal)
    )

    executed = executor.execute_dashboard_motion(
        prepared["plan_id"],
        "cross-negative-pi",
    )

    assert executed["primitive_success"] is True
    assert executed["stop_reason"] == "reached"
    assert executed["metrics"]["certificate_verified_before_first_action"] is True
    assert len(env.calls) > 0


def test_discard_prepared_motion_invalidates_all_descendants():
    executor, _env = _executor(_NavigationBackend())
    first = executor.prepare_dashboard_motion("chassis", "forward")
    second = executor.prepare_dashboard_motion(
        "chassis",
        "turn_left",
        predecessor_plan_id=first["plan_id"],
    )
    third = executor.prepare_dashboard_motion(
        "chassis",
        "backward",
        predecessor_plan_id=second["plan_id"],
    )

    result = executor.discard_dashboard_motion(first["plan_id"])

    assert result["discarded"] is True
    assert result["invalidated_descendant_plan_ids"] == [
        second["plan_id"],
        third["plan_id"],
    ]
    for plan, command in ((second, "second"), (third, "third")):
        with pytest.raises(RuntimeError, match="not executable"):
            executor.execute_dashboard_motion(plan["plan_id"], command)


def test_prepared_dashboard_queue_rejects_one_shots_and_unsupported_torso():
    executor, _env = _executor(_NavigationBackend())

    for action in ("observe", "open", "close"):
        with pytest.raises(ValueError, match="one-shot"):
            executor.prepare_dashboard_motion("left_arm", action)
    with pytest.raises(RuntimeError, match="torso_control_unsupported"):
        executor.prepare_dashboard_motion("chassis", "up")


def test_prepared_arm_successor_uses_predicted_eef_and_has_no_extra_hold():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.prepared_calls = []

        def plan_whole_body_trajectory(
            self,
            *,
            start_q=None,
            start_eef_pose=None,
            background=False,
            **kwargs,
        ):
            self.prepared_calls.append(
                {
                    **kwargs,
                    "start_q": np.asarray(start_q).copy(),
                    "start_eef_pose": (
                        np.asarray(start_eef_pose[0]).copy(),
                        np.asarray(start_eef_pose[1]).copy(),
                    ),
                    "background": background,
                }
            )
            return super().plan_whole_body_trajectory(**kwargs)

    backend = Backend()
    executor, env = _executor(backend)

    first = executor.prepare_dashboard_motion("left_arm", "forward")
    second = executor.prepare_dashboard_motion(
        "left_arm",
        "up",
        predecessor_plan_id=first["plan_id"],
    )
    first_call, second_call = backend.prepared_calls
    np.testing.assert_allclose(first_call["target_xyz"], [0.53, 0.0, 0.0])
    np.testing.assert_allclose(second_call["start_eef_pose"][0], [0.53, 0.0, 0.0])
    np.testing.assert_allclose(second_call["target_xyz"], [0.53, 0.0, 0.03])
    np.testing.assert_allclose(
        second_call["start_q"],
        first["predicted_terminal"]["joint_positions"],
    )
    assert second["predecessor_plan_id"] == first["plan_id"]

    first_plan = executor._prepared_motions[first["plan_id"]]["plan"]
    first_certificate = first_plan["whole_body_certificate"]
    backend.target = np.asarray(first_call["target_xyz"], dtype=np.float64)
    backend.target_quat = np.asarray(
        first_call["target_quat_xyzw"], dtype=np.float64
    )
    backend.execution_eef_positions = np.asarray(
        first_certificate["selected_eef_execution_positions"],
        dtype=np.float32,
    )
    backend.execution_eef_quaternions = np.asarray(
        first_certificate["selected_eef_execution_quaternions_xyzw"],
        dtype=np.float32,
    )
    backend.dense_eef_positions = np.asarray(
        first_certificate["selected_eef_dense_positions"],
        dtype=np.float32,
    )
    backend.dense_eef_quaternions = np.asarray(
        first_certificate["selected_eef_dense_quaternions_xyzw"],
        dtype=np.float32,
    )
    executed = executor.execute_dashboard_motion(first["plan_id"], "arm-command")
    assert executed["primitive_success"] is True
    assert len(env.calls) == len(backend.execution_eef_positions)
    guard = executed["metrics"]["whole_body_eef_path_guard"]
    assert guard["terminal_commands_sent"] == 1
    assert executed["metrics"]["live_start_checked"] is True
    assert executed["metrics"]["live_start_equality_skipped"] is False
    assert executed["metrics"]["prepared_start_strict_float32_equal"] is True
    assert executed["metrics"]["collision_revalidation_skipped"] is True


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
    assert len(backend.whole_body_plan_calls) > 1
    assert any(
        round_report.get("execution_stop_reason")
        == "whole_body_replan_checkpoint"
        for round_report in result["metrics"]["replan_rounds"]
    )
    assert {
        call["search_profile"] for call in backend.whole_body_plan_calls
    } == {WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG}


@_REQUIRES_TORCH
@pytest.mark.parametrize("local_ik_success", [True, False])
def test_dashboard_jog_profile_uses_local_ik_then_falls_back_and_stops_first_safe(
    tmp_path,
    local_ik_success,
):
    torch = pytest.importorskip("torch")
    joint_names = tuple(_FakeBackend().joint_names)
    live_q = np.full(len(joint_names), -0.25, dtype=np.float32)
    start_q = np.linspace(
        0.0,
        0.027,
        len(joint_names),
        dtype=np.float32,
    )
    start_xyz = np.asarray([0.5, 0.0, 0.0], dtype=np.float64)
    target_xyz = np.asarray([0.53, 0.0, 0.0], dtype=np.float64)

    class Robot:
        joints = dict.fromkeys(joint_names)
        links = {"left_eef_link": object(), "right_eef_link": object()}

        @staticmethod
        def get_joint_positions():
            return live_q.copy()

    rollout_retract = torch.zeros(21, dtype=torch.float32)
    cspace_retract = torch.zeros(21, dtype=torch.float32)
    rollout = SimpleNamespace(
        dynamics_model=SimpleNamespace(retract_config=rollout_retract)
    )
    kinematics = SimpleNamespace(
        joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
        kinematics_config=SimpleNamespace(
            cspace=SimpleNamespace(retract_config=cspace_retract)
        ),
    )
    motion_gen = SimpleNamespace(
        kinematics=kinematics,
        get_all_rollout_instances=lambda: [rollout],
    )

    class Generator:
        batch_size = 2
        robot_joint_names = list(joint_names)
        mg = {"default": motion_gen}

        def __init__(self):
            self.compute_calls = []
            self.collision_calls = []
            self.world_updates = 0

        def update_obstacles(self):
            self.world_updates += 1
            self._rpent_obstacle_refresh_metrics = {
                "clock": "time.monotonic",
                "completed_monotonic_ns": self.world_updates,
                "count": self.world_updates,
                "mode": "pose_only",
                "elapsed_s": 0.001,
                "fallback": False,
            }

        def compute_trajectories(self, _positions, _quaternions, **kwargs):
            policy = getattr(self, "_rpent_plan_override", None)
            graph = bool(policy["enable_graph"]) if policy is not None else False
            self.compute_calls.append(
                {
                    "graph": graph,
                    "policy": None if policy is None else dict(policy),
                    "kwargs": dict(kwargs),
                }
            )
            if kwargs["ik_only"]:
                if local_ik_success:
                    return (
                        np.asarray([True, False]),
                        [SimpleNamespace(index=0), None],
                    )
                return np.asarray([False, False]), [None, None]
            if not graph:
                return np.asarray([False, False]), [None, None]
            return (
                np.asarray([True, True]),
                [SimpleNamespace(index=0), SimpleNamespace(index=1)],
            )

        def check_collisions(self, q_trajectory, **kwargs):
            q = np.asarray(q_trajectory, dtype=np.float32)
            self.collision_calls.append((q.copy(), dict(kwargs)))
            return np.zeros(len(q), dtype=bool)

    generator = Generator()
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._torch = torch
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default")
    backend._generator = lambda **_kwargs: generator
    backend._find_robot = lambda: Robot()
    backend._eef_link_name = lambda _robot, hand: f"{hand}_eef_link"
    backend.get_eef_pose = lambda _hand: (
        start_xyz.copy(),
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )
    backend._all_attached_objects = lambda **_kwargs: (
        None,
        {"left": None, "right": None},
    )
    config_path = tmp_path / "whole_body.yaml"
    config_path.write_text("kind: whole_body_test\n", encoding="utf-8")
    backend._whole_body_config_path = lambda _hand: config_path

    def merge(_generator, _robot, path, *, start_q):
        goal = np.asarray(start_q, dtype=np.float32).copy()
        goal[6 + int(path.index)] = 0.03
        return goal.reshape(1, -1), {
            "source": "dashboard_profile_test",
            "candidate": int(path.index),
        }

    def eef_poses(_generator, q_trajectory):
        count = len(np.asarray(q_trajectory))
        return (
            np.linspace(start_xyz, target_xyz, count, dtype=np.float64),
            np.repeat(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                count,
                axis=0,
            ),
        )

    backend._whole_body_path_to_full_joint_trajectory = merge
    backend._curobo_eef_poses = eef_poses

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target_xyz,
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=99.0,
        search_profile=WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
        start_q=start_q,
        start_eef_pose=(
            start_xyz,
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ),
    )

    assert result["ok"] is True
    assert generator.compute_calls[0]["policy"] is None
    assert generator.compute_calls[0]["kwargs"]["ik_only"] is True
    assert generator.compute_calls[0]["kwargs"]["is_local"] is False
    assert generator.compute_calls[0]["kwargs"]["max_attempts"] == 1
    assert generator.compute_calls[0]["kwargs"]["ik_fail_return"] == 1
    assert generator.compute_calls[0]["kwargs"]["enable_finetune_trajopt"] is False
    assert generator.compute_calls[0]["kwargs"]["finetune_attempts"] == 0
    assert generator.compute_calls[0]["kwargs"]["timeout"] <= (
        WHOLE_BODY_DASHBOARD_JOG_LOCAL_IK_DEADLINE_S
    )
    np.testing.assert_allclose(
        np.asarray(
            generator.compute_calls[0]["kwargs"]["initial_joint_pos"]
        ),
        start_q,
    )
    active_expected = np.asarray(
        [
            start_q[joint_names.index(name)]
            for name in WHOLE_BODY_ACTIVE_JOINT_NAMES
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(rollout_retract), active_expected)
    np.testing.assert_allclose(np.asarray(cspace_retract), active_expected)
    if local_ik_success:
        assert len(generator.compute_calls) == 1
        expected_stage = "local_ik"
    else:
        assert [call["graph"] for call in generator.compute_calls] == [
            False,
            False,
            True,
        ]
        for call in generator.compute_calls[1:]:
            assert call["kwargs"]["ik_only"] is False
            assert call["kwargs"]["max_attempts"] == 3
            assert call["kwargs"]["ik_fail_return"] == 3
            assert call["kwargs"]["finetune_attempts"] == 1
        assert generator.compute_calls[1]["kwargs"]["timeout"] <= (
            WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
        )
        assert generator.compute_calls[2]["kwargs"]["timeout"] <= (
            WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
        )
        expected_stage = "graph_trajopt"
    assert len(generator.collision_calls) == 1
    assert len(result["metrics"]["candidate_audit"]) == 1
    assert result["metrics"]["candidate_audit"][0]["certified"] is True
    assert result["metrics"]["selected_solver_stage"] == expected_stage
    assert result["whole_body_certificate"]["start_q_sha256"] == hashlib.sha256(
        np.ascontiguousarray(start_q, dtype=np.float32).tobytes()
    ).hexdigest()
    policy = result["metrics"]["planning_policy"]
    assert policy["search_profile"] == WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
    assert policy["graph_fallback"] is True
    assert policy["candidate_certification"] == "first_fully_certified"
    assert result["metrics"]["solver_stages"][-1][
        "certification_short_circuit"
    ]["remaining_successes_not_certified"] == (0 if local_ik_success else 1)


def test_public_move_to_keeps_default_whole_body_search_profile():
    backend = _FakeBackend()
    executor, _env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.5, 0.0, 0.0],
        plan_only=True,
    )

    assert result["primitive_success"] is True
    assert len(backend.whole_body_plan_calls) == 1
    assert backend.whole_body_plan_calls[0]["search_profile"] == (
        WHOLE_BODY_SEARCH_PROFILE_DEFAULT
    )


def test_public_move_to_replans_only_after_a_certified_progress_checkpoint():
    backend = _FakeBackend()
    executor, env = _executor(backend)

    result = executor.move_to(
        hand="left",
        target_xyz=[0.506, 0.0, 0.0],
        position_tolerance_m=0.001,
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(backend.whole_body_plan_calls) == 2
    assert len(env.calls) == 37
    assert [
        round_report.get("execution_stop_reason")
        for round_report in result["metrics"]["replan_rounds"]
    ] == [
        "whole_body_replan_checkpoint",
        "reached",
    ]
    checkpoint = result["metrics"]["replan_rounds"][0][
        "whole_body_replan_checkpoint"
    ]
    assert checkpoint["dynamics_gate_used"] is False
    assert result["metrics"]["env_actions_sent"] == len(env.calls)


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
    assert len(backend.planned_targets) > 3
    np.testing.assert_allclose(
        backend.planned_targets[:2],
        np.asarray([[0.53, 0.0, 0.0], [0.53, 0.0, 0.0]]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        backend.planned_targets[2:],
        np.tile([0.53, 0.0025, 0.0], (len(backend.planned_targets) - 2, 1)),
        atol=1e-9,
    )
    assert len(env.calls) > 0


def test_jog_eef_compensation_candidates_share_twelve_second_budget():
    backend = _FakeBackend()
    executor, env = _executor(backend)
    budgets = []
    calls = []

    def move_with_budget(**kwargs):
        calls.append(dict(kwargs))
        budgets.append(float(kwargs["planning_budget_s"]))
        return {
            "primitive_success": False,
            "task_success": False,
            "stop_reason": "unreachable",
            "recoverable": True,
            "metrics": {
                "planning_spent_s": 12.0,
                "execution_spent_s": 0.0,
                "env_actions_sent": 0,
            },
            "diagnostics": {},
        }

    executor._move_to_whole_body_impl = move_with_budget

    result = executor.jog_eef("left", "forward", timeout_s=240.0)

    assert budgets == pytest.approx([12.0])
    assert calls[0]["search_profile"] == (
        WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
    )
    assert calls[0]["replan_checkpoint_position_improvement_m"] == pytest.approx(
        WHOLE_BODY_DASHBOARD_JOG_REPLAN_POSITION_IMPROVEMENT_M
    )
    assert result["primitive_success"] is False
    assert result["stop_reason"] == "planning_budget_exhausted"
    assert result["metrics"]["candidate_attempts"][-1][
        "remaining_planning_budget_s"
    ] == pytest.approx(0.0)
    assert env.calls == []


def test_jog_eef_candidate_feedback_reads_are_inside_total_wall_deadline(
    monkeypatch,
):
    operations = []

    @contextmanager
    def deadline(timeout_s, operation):
        operations.append((str(operation), float(timeout_s)))
        if operation == "EEF jog candidate transaction":
            raise TimeoutError("blocked candidate feedback")
        yield

    monkeypatch.setattr(
        "robots.behavior.planner_executor._wall_clock_deadline",
        deadline,
    )
    backend = _FakeBackend()
    executor, env = _executor(backend)
    private_calls = []
    executor._move_to_whole_body_impl = lambda **kwargs: private_calls.append(
        kwargs
    )

    result = executor.jog_eef("left", "forward", timeout_s=999.0)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "timeout"
    assert private_calls == []
    assert operations[0] == (
        "EEF jog call-start pose transaction",
        pytest.approx(12.0),
    )
    assert operations[1][0] == "EEF jog candidate transaction"
    assert 0.0 < operations[1][1] <= 12.0
    assert env.calls == []


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


def test_jog_wrist_nonfinite_final_position_cannot_preserve_success():
    class Backend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.pose_reads = 0

        @staticmethod
        def wrist_visual_rotation_capability(selected_hand):
            return {
                "verified": True,
                "hand": selected_hand,
                "clockwise_angle_sign": 1.0,
                "probe_artifact": "left-visual-probe.json",
            }

        def get_eef_pose(self, hand):
            del hand
            self.pose_reads += 1
            if self.pose_reads == 1:
                return self.pose.copy(), self.quat.copy()
            return (
                np.asarray([float("nan"), 0.0, 0.0]),
                self.quat.copy(),
            )

    backend = Backend()
    executor, _env = _executor(backend)
    _add_wrist_frame(executor, "left")
    executor.move_to = lambda **_kwargs: {
        "primitive_success": True,
        "task_success": False,
        "stop_reason": "reached",
        "recoverable": True,
        "metrics": {"env_actions_sent": 1},
        "diagnostics": {},
    }

    result = executor.jog_wrist("left", "rotate_left", timeout_s=10.0)

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "pose_feedback_unavailable"
    assert result["metrics"]["partial_motion"] is True
    assert result["metrics"]["env_actions_sent"] == 1


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
    np.testing.assert_array_equal(
        backend.navigation_targets,
        backend.q_path,
    )
    terminal = result["metrics"]["navigation_terminal"]
    assert terminal["commands_sent"] == 1
    assert terminal["command_limit"] == TERMINAL_COMMAND_LIMIT
    feedback_latency = result["metrics"]["navigation_feedback_latency"]
    assert feedback_latency["clock"] == "time.monotonic"
    assert set(feedback_latency["phases"]) == {
        "attachment_identity",
        "navigation_isolation",
        "whole_body_contact",
        "joint_tracking",
        "base_pose",
    }
    assert all(
        phase["count"] == len(backend.q_path)
        and phase["total_s"] >= 0.0
        and phase["max_s"] >= 0.0
        for phase in feedback_latency["phases"].values()
    )
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


def test_navigation_attachment_mismatch_records_completed_feedback_latency_phase():
    class Backend(_NavigationBackend):
        def __init__(self):
            super().__init__()
            self.changed_attachment = object()

        def get_attached_object(self, hand):
            if self.env.calls and hand == "left":
                return {"left_eef_link": self.changed_attachment}
            return None

    backend = Backend()
    executor, env = _executor(backend)

    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert len(env.calls) == 1
    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_identity_mismatch"
    phase = result["metrics"]["navigation_feedback_latency"]["phases"][
        "attachment_identity"
    ]
    assert phase["count"] == 1
    assert phase["total_s"] >= 0.0
    assert phase["max_s"] >= 0.0


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
    ("certificate_field", "tampered_value"),
    [
        ("terminal_command_limit", TERMINAL_COMMAND_LIMIT + 1),
        ("terminal_position_tolerance_m", -1.0),
        ("terminal_orientation_tolerance_rad", -1.0),
    ],
)
def test_navigation_terminal_policy_certificate_tamper_executes_zero_actions(
    certificate_field,
    tampered_value,
):
    backend = _NavigationBackend()
    unsafe_plan = backend._navigation_plan()
    unsafe_plan["metrics"]["base_trajectory_certificate"][
        certificate_field
    ] = tampered_value
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


def test_navigation_runtime_never_reads_dynamics_reporter():
    backend = _NavigationDynamicsBackend([])
    executor, env = _executor(backend)

    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.calls) == len(backend.q_path)
    assert backend.dynamics_calls == 0


@pytest.mark.parametrize("success_on_terminal_command", [6, None])
def test_navigation_terminal_has_six_command_total_cap_without_seventh(
    success_on_terminal_command,
):
    class Backend(_NavigationBackend):
        def __init__(self):
            super().__init__()
            self.terminal_commands = 0

        def joint_target_to_action(self, target_q, *, hand, fixed_reference):
            action = super().joint_target_to_action(
                target_q,
                hand=hand,
                fixed_reference=fixed_reference,
            )
            if np.array_equal(np.asarray(target_q), self.q_path[-1]):
                self.terminal_commands += 1
                if (
                    success_on_terminal_command is None
                    or self.terminal_commands < success_on_terminal_command
                ):
                    self.base_pose = self.q_path[-2, [0, 1, 5]].astype(
                        np.float64
                    )
            return action

    backend = Backend()
    executor, env = _executor(backend)
    result = executor.navigate_to(
        target_xyz=[1.0, 0.0, 0.0],
        standoff_m=0.85,
        max_travel_m=1.0,
        timeout_s=5.0,
    )

    assert backend.terminal_commands == TERMINAL_COMMAND_LIMIT
    assert len(env.calls) == len(backend.q_path) - 1 + TERMINAL_COMMAND_LIMIT
    assert result["metrics"]["navigation_terminal"]["commands_sent"] == (
        TERMINAL_COMMAND_LIMIT
    )
    if success_on_terminal_command is None:
        assert result["primitive_success"] is False
        assert result["stop_reason"] == "target_tolerance_not_met"
    else:
        assert result["primitive_success"] is True
        assert result["stop_reason"] == "reached"


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
