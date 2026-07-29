from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import CameraIntrinsics, FrameCache
from robots.behavior.planner_executor import (
    DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    DASHBOARD_EXECUTION_MAX_WAYPOINTS,
    CuroboPlanningError,
    PlannerExecutor,
    RealCuroboBackend,
    _deterministic_execution_waypoints,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT,
    DASHBOARD_HOLD_ARM_DELAY_S,
    DASHBOARD_PREDICTED_PLAN_DEPTH,
    ENV_ACTION_SEGMENTS,
)


class _LifecycleEnv:
    def __init__(self, receipts=()) -> None:
        self.receipts = deque(receipts)
        self.actions: list[np.ndarray] = []
        self._last_info = {"done": {"success": False}}
        self._official_success_latched = False
        self.on_action = None

    def planner_step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        if callable(self.on_action):
            self.on_action(action)
        receipt = (
            self.receipts.popleft()
            if self.receipts
            else {
                "raw_success": False,
                "terminated": False,
                "truncated": False,
            }
        )
        if isinstance(receipt, BaseException):
            raise receipt
        info = {"done": {"success": bool(receipt.get("raw_success"))}}
        terminal_capture = receipt.get("terminal_capture")
        if isinstance(terminal_capture, dict):
            info["_rpent"] = {"terminal_capture": terminal_capture}
        self._last_info = info
        return (
            None,
            0.0,
            bool(receipt.get("terminated")),
            bool(receipt.get("truncated")),
            info,
        )


class _ActionBackend:
    def __init__(self) -> None:
        self.converted: list[np.ndarray] = []
        self.eef_pose_reads: list[str] = []
        self.live_q = np.zeros((28,), dtype=np.float32)
        self.control_targets: list[np.ndarray] = []
        self.control_start_q: list[np.ndarray] = []
        self.control_live_after: list[np.ndarray] = []

    def joint_target_to_action(self, target_q, *, hand=None):
        del hand
        q = np.asarray(target_q, dtype=np.float32).reshape(-1)
        self.converted.append(q.copy())
        action = np.zeros((ACTION_DIM,), dtype=np.float32)
        action[0] = q[0]
        return action

    def get_eef_pose(self, hand):
        self.eef_pose_reads.append(str(hand))
        return (
            np.zeros((3,), dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )

    def action_for_control_target(self, target_q):
        target = np.asarray(target_q, dtype=np.float32).reshape(28)
        self.control_targets.append(target.copy())
        self.control_start_q.append(self.live_q.copy())
        action = np.zeros((ACTION_DIM,), dtype=np.float32)
        action[ENV_ACTION_SEGMENTS["base"]] = target[[0, 1, 5]] - self.live_q[[0, 1, 5]]
        action[ENV_ACTION_SEGMENTS["trunk"]] = target[6:10]
        action[ENV_ACTION_SEGMENTS["left_arm"]] = target[10:17]
        action[ENV_ACTION_SEGMENTS["right_arm"]] = target[17:24]
        action[ENV_ACTION_SEGMENTS["left_gripper"]] = target[24]
        action[ENV_ACTION_SEGMENTS["right_gripper"]] = target[25]
        return action

    def apply_action(self, action):
        command = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        self.live_q[[0, 1, 5]] += command[ENV_ACTION_SEGMENTS["base"]] * 0.5
        self.control_live_after.append(self.live_q.copy())


def _executor(tmp_path, *, receipts=()):
    env = _LifecycleEnv(receipts)
    backend = _ActionBackend()
    env.on_action = backend.apply_action
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )
    return executor, env, backend


class _Rollout:
    primitive_collision_cost = None
    primitive_collision_constraint = None
    robot_self_collision_cost = None
    robot_self_collision_constraint = None


class _MotionGenerator:
    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.kinematics = SimpleNamespace(joint_names=joint_names)

    @staticmethod
    def get_all_rollout_instances():
        return [_Rollout()]


class _FakeGenerator:
    def __init__(
        self,
        *,
        successes: list[bool],
        rows: np.ndarray | None = None,
        includes_start_state: bool = False,
    ) -> None:
        self.batch_size = len(successes)
        self.robot_joint_names = ("j0", "j1")
        self.mg = {"default": _MotionGenerator(self.robot_joint_names)}
        self.successes = successes
        self.rows = (
            np.asarray(rows, dtype=np.float32)
            if rows is not None
            else np.zeros((1, 2), dtype=np.float32)
        )
        self.includes_start_state = includes_start_state
        self.kwargs = None

    def compute_trajectories(self, *_args, **kwargs):
        self.kwargs = kwargs
        path = SimpleNamespace(
            joint_names=self.robot_joint_names,
            position=self.rows,
        )
        return SimpleNamespace(
            success=np.asarray(self.successes, dtype=bool),
            status="converged" if any(self.successes) else "ik_fail",
            get_paths=lambda: [path for _ in self.successes],
        )


def test_curobo_internal_goal_convergence_and_one_second_solver_contract():
    backend = RealCuroboBackend(None)
    generator = _FakeGenerator(successes=[False])
    robot = SimpleNamespace(joints={"j0": object(), "j1": object()})

    result = backend._goal_only_curobo_trajectory(
        generator=generator,
        robot=robot,
        target_positions=object(),
        target_quaternions=object(),
        start_q=np.asarray([0.1, -0.2], dtype=np.float32),
        embodiment="default",
        position_only=False,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "unreachable"
    assert result["metrics"] == {
        "solver": "fast_trajopt",
        "solver_timeout_s": 1.0,
        "solver_elapsed_s": result["metrics"]["solver_elapsed_s"],
        "enable_graph": False,
        "max_attempts": 1,
        "successes": [False],
        "curobo_statuses": result["metrics"]["curobo_statuses"],
    }
    assert generator.kwargs["timeout"] == DASHBOARD_CUROBO_PLAN_TIMEOUT_S
    assert generator.kwargs["max_attempts"] == 1
    assert generator.kwargs["enable_finetune_trajopt"] is True
    assert generator.kwargs["ik_world_collision_check"] is False
    np.testing.assert_allclose(
        generator.kwargs["initial_joint_pos"].detach().cpu().numpy(),
        [0.1, -0.2],
    )


def test_curobo_success_removes_explicit_start_and_keeps_at_most_three_commands():
    rows = np.arange(12, dtype=np.float32).reshape(6, 2)
    generator = _FakeGenerator(
        successes=[True],
        rows=rows,
        includes_start_state=True,
    )
    backend = RealCuroboBackend(None)
    backend.joint_trajectory_to_actions = lambda trajectory, start_q: np.zeros(
        (len(trajectory), ACTION_DIM), dtype=np.float32
    )
    robot = SimpleNamespace(joints={"j0": object(), "j1": object()})

    result = backend._goal_only_curobo_trajectory(
        generator=generator,
        robot=robot,
        target_positions=object(),
        target_quaternions=object(),
        start_q=np.asarray([0.0, 0.0], dtype=np.float32),
        embodiment="default",
        position_only=True,
    )

    assert result["ok"] is True
    assert result["metrics"]["start_state_row_removed"] is True
    assert result["metrics"]["geometric_waypoints"] == 3
    np.testing.assert_array_equal(result["joint_trajectory"][-1], rows[-1])


@pytest.mark.parametrize("count", [1, 2, 3, 4, 20])
def test_deterministic_waypoint_sampling_is_ordered_bounded_and_contains_terminal(
    count,
):
    trajectory = np.arange(count * 2, dtype=np.float32).reshape(count, 2)

    selected, indices = _deterministic_execution_waypoints(trajectory)

    assert 1 <= len(selected) <= DASHBOARD_EXECUTION_MAX_WAYPOINTS
    assert indices == sorted(set(indices))
    assert indices[-1] == count - 1
    np.testing.assert_array_equal(selected[-1], trajectory[-1])


def test_zero_executable_waypoints_is_a_planning_failure():
    with pytest.raises(ValueError, match="must be \\[T,D\\]"):
        _deterministic_execution_waypoints(np.zeros((0, 28), dtype=np.float32))


@pytest.mark.parametrize("hand", ["left", "right"])
def test_wrist_visual_axis_uses_same_step_camera_and_live_eef_pose(
    tmp_path,
    hand,
):
    executor, env, backend = _executor(tmp_path)
    env._env_steps = 0
    intrinsics = CameraIntrinsics(
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        width=2,
        height=2,
    )
    executor.frame_cache.add_capture_group(
        frames={
            name: {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_m": np.ones((2, 2), dtype=np.float32),
                "intrinsics": intrinsics,
                "camera_to_world": np.eye(4, dtype=np.float64),
            }
            for name in ("head", "left_wrist", "right_wrist")
        },
        step_index=0,
        capture_group_id="capture:0:test",
        timestamp_s=-1.0,
    )

    calibration = executor._wrist_camera_rotation_calibration(hand)

    assert calibration["verified"] is True
    assert calibration["visual_ccw_angle_sign"] == 1.0
    np.testing.assert_allclose(
        calibration["screen_normal_axis_eef"],
        [0.0, 0.0, 1.0],
    )
    assert backend.eef_pose_reads == [hand]


def test_wrist_visual_axis_rejects_a_capture_from_an_older_env_step(tmp_path):
    executor, env, backend = _executor(tmp_path)
    env._env_steps = 1
    intrinsics = CameraIntrinsics(
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        width=2,
        height=2,
    )
    executor.frame_cache.add_capture_group(
        frames={
            name: {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_m": np.ones((2, 2), dtype=np.float32),
                "intrinsics": intrinsics,
                "camera_to_world": np.eye(4, dtype=np.float64),
            }
            for name in ("head", "left_wrist", "right_wrist")
        },
        step_index=0,
        capture_group_id="capture:0:old",
        timestamp_s=-1.0,
    )

    calibration = executor._wrist_camera_rotation_calibration("left")

    assert calibration["available"] is False
    assert calibration["verified"] is False
    assert "current simulator step" in calibration["reason"]
    assert backend.eef_pose_reads == []


@pytest.mark.parametrize("geometric_waypoints", [1, 2, 3])
def test_executor_preserves_base_budget_and_holds_absolute_targets_for_five_cycles(
    tmp_path,
    geometric_waypoints,
):
    executor, env, backend = _executor(tmp_path)
    trajectory = np.zeros((geometric_waypoints, ACTION_DIM), dtype=np.float32)
    joint_targets = np.zeros((geometric_waypoints, 28), dtype=np.float32)
    for index, target in enumerate(joint_targets, start=1):
        target[[0, 1, 5]] = [index * 0.05, index * -0.02, index * 0.087]
        target[6:10] = index * 100
        target[10:17] = index * 200
        target[17:24] = index * 300
        target[24:26] = 1

    result = executor._execute_goal_only_waypoints(
        actions=trajectory,
        joint_targets=joint_targets,
    )

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert result["metrics"]["geometric_waypoints"] == geometric_waypoints
    assert result["metrics"]["control_cycles_per_waypoint"] == 5
    assert result["metrics"]["env_actions_sent"] == geometric_waypoints * 5
    assert result["metrics"]["executed_control_cycles"] == geometric_waypoints * 5
    for index, geometric_target in enumerate(joint_targets):
        cycle_actions = env.actions[index * 5 : (index + 1) * 5]
        cycle_targets = backend.control_targets[index * 5 : (index + 1) * 5]
        cycle_starts = backend.control_start_q[index * 5 : (index + 1) * 5]
        assert len(cycle_actions) == 5
        assert len(cycle_targets) == 5
        for cycle_action, cycle_target, cycle_start in zip(
            cycle_actions,
            cycle_targets,
            cycle_starts,
            strict=True,
        ):
            cycle_action = cycle_action.reshape(-1)
            np.testing.assert_array_equal(cycle_target, geometric_target)
            np.testing.assert_allclose(
                cycle_action[ENV_ACTION_SEGMENTS["base"]],
                geometric_target[[0, 1, 5]] - cycle_start[[0, 1, 5]],
            )
            np.testing.assert_array_equal(
                cycle_action[ENV_ACTION_SEGMENTS["trunk"]],
                geometric_target[6:10],
            )
            np.testing.assert_array_equal(
                cycle_action[ENV_ACTION_SEGMENTS["left_arm"]],
                geometric_target[10:17],
            )
            np.testing.assert_array_equal(
                cycle_action[ENV_ACTION_SEGMENTS["right_arm"]],
                geometric_target[17:24],
            )
        first_live = cycle_starts[0][[0, 1, 5]]
        final_live = backend.control_live_after[(index + 1) * 5 - 1][[0, 1, 5]]
        target_base = geometric_target[[0, 1, 5]]
        assert np.linalg.norm(target_base - final_live) < np.linalg.norm(
            target_base - first_live
        )
        assert np.all((target_base - final_live) * (target_base - first_live) >= 0)


@pytest.mark.parametrize(("opening", "expected"), [(1.0, 1.0), (0.0, -1.0)])
def test_gripper_command_uses_goal_only_executor(tmp_path, opening, expected):
    executor, env, _backend = _executor(tmp_path)

    result = executor._gripper_command("left", opening=opening)

    assert result["primitive_success"] is True
    assert result["stop_reason"] == "reached"
    assert len(env.actions) == 5
    for action in env.actions:
        np.testing.assert_array_equal(
            action[ENV_ACTION_SEGMENTS["left_gripper"]],
            expected,
        )


@pytest.mark.parametrize(
    ("receipt", "expected_success", "expected_stop"),
    [
        (
            {
                "raw_success": True,
                "terminated": True,
                "truncated": True,
                "terminal_capture": {"capture_group_id": "terminal"},
            },
            True,
            "official_task_success",
        ),
        (
            {"raw_success": False, "terminated": True, "truncated": True},
            False,
            "environment_terminated",
        ),
        (
            {"raw_success": False, "terminated": False, "truncated": True},
            False,
            "environment_truncated",
        ),
    ],
)
def test_raw_success_terminated_and_truncated_stop_remaining_waypoints(
    tmp_path,
    receipt,
    expected_success,
    expected_stop,
):
    executor, env, backend = _executor(
        tmp_path,
        receipts=[
            {"raw_success": False, "terminated": False, "truncated": False},
            receipt,
        ],
    )

    result = executor._execute_goal_only_waypoints(
        actions=np.zeros((3, ACTION_DIM), dtype=np.float32),
        joint_targets=np.zeros((3, 28), dtype=np.float32),
    )

    assert result["task_success"] is expected_success
    assert result["stop_reason"] == expected_stop
    assert result["metrics"]["geometric_waypoints"] == 3
    assert result["metrics"]["control_cycles_per_waypoint"] == 5
    assert result["metrics"]["env_actions_sent"] == 2
    assert result["metrics"]["executed_control_cycles"] == 2
    assert len(env.actions) == 2
    assert len(backend.control_targets) == 2
    if expected_success:
        assert result["terminal_capture"]["capture_group_id"] == "terminal"


def test_rpc_failure_is_infrastructure_error_and_stops_remaining_waypoints(tmp_path):
    executor, env, backend = _executor(
        tmp_path,
        receipts=[
            {"raw_success": False, "terminated": False, "truncated": False},
            ConnectionError("env process exited"),
        ],
    )

    result = executor._execute_goal_only_waypoints(
        actions=np.zeros((3, ACTION_DIM), dtype=np.float32),
        joint_targets=np.zeros((3, 28), dtype=np.float32),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "rpc_error"
    assert result["metrics"]["geometric_waypoints"] == 3
    assert result["metrics"]["control_cycles_per_waypoint"] == 5
    assert result["metrics"]["env_actions_sent"] == 2
    assert result["metrics"]["executed_control_cycles"] == 1
    assert len(env.actions) == 2
    assert len(backend.control_targets) == 2
    assert "ConnectionError" in result["diagnostics"]["error"]


class _PreparedBackend(_ActionBackend):
    def __init__(self, *, failure: dict | None = None) -> None:
        super().__init__()
        self.live_q_reads = 0
        self.live_base_reads = 0
        self.plan_calls: list[dict] = []
        self.failure = failure

    def get_joint_positions(self):
        self.live_q_reads += 1
        return np.zeros((28,), dtype=np.float32)

    def get_base_pose(self):
        self.live_base_reads += 1
        return np.zeros((3,), dtype=np.float64)

    def plan_relative_navigation_trajectory(self, **kwargs):
        self.plan_calls.append(kwargs)
        if self.failure is not None:
            return dict(self.failure)
        start = np.asarray(kwargs["start_q"], dtype=np.float32).copy()
        terminal = start.copy()
        terminal[0] += 0.05
        return {
            "ok": True,
            "joint_trajectory": np.stack([terminal], axis=0),
            "action_trajectory": np.zeros((1, ACTION_DIM), dtype=np.float32),
            "base_goal": np.asarray([terminal[0], terminal[1], terminal[5]]),
            "metrics": {"solver": "fast_trajopt"},
        }

    @staticmethod
    def whole_body_eef_poses(hand, q_trajectory):
        del hand
        count = len(np.asarray(q_trajectory))
        return (
            np.zeros((count, 3), dtype=np.float64),
            np.repeat([[0.0, 0.0, 0.0, 1.0]], count, axis=0),
        )

    @staticmethod
    def torso_poses(q_trajectory):
        count = len(np.asarray(q_trajectory))
        return (
            np.zeros((count, 3), dtype=np.float64),
            np.repeat([[0.0, 0.0, 0.0, 1.0]], count, axis=0),
        )


def test_first_plan_reads_live_state_and_predicted_successor_uses_prior_terminal(
    tmp_path,
):
    env = _LifecycleEnv()
    backend = _PreparedBackend()
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )

    first = executor.prepare_dashboard_motion("chassis", "forward")
    second = executor.prepare_dashboard_motion(
        "chassis",
        "forward",
        predecessor_plan_id=first["plan_id"],
        background=True,
    )

    assert backend.live_q_reads == 1
    assert backend.live_base_reads == 1
    assert len(backend.plan_calls) == 2
    np.testing.assert_allclose(
        backend.plan_calls[1]["start_q"],
        first["predicted_terminal"]["joint_positions"],
    )
    assert backend.plan_calls[0]["timeout_s"] == 1.0
    assert backend.plan_calls[1]["timeout_s"] == 1.0
    assert second["predecessor_plan_id"] == first["plan_id"]


def test_curobo_failure_classification_is_preserved_by_prepared_planner(tmp_path):
    failure = {
        "ok": False,
        "stop_reason": "unreachable",
        "error": "CuRobo failed to converge within 1.0 s",
    }
    executor = PlannerExecutor(
        env=_LifecycleEnv(),
        frame_cache=FrameCache(),
        backend=_PreparedBackend(failure=failure),
        output_dir=tmp_path,
    )

    with pytest.raises(CuroboPlanningError) as raised:
        executor.prepare_dashboard_motion("chassis", "forward")

    assert raised.value.stop_reason == "unreachable"
    assert "1.0 s" in str(raised.value)


def test_frozen_low_latency_constants():
    assert DASHBOARD_CUROBO_PLAN_TIMEOUT_S == 1.0
    assert DASHBOARD_EXECUTION_MAX_WAYPOINTS == 3
    assert DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT == 5
    assert DASHBOARD_PREDICTED_PLAN_DEPTH == 20
    assert DASHBOARD_HOLD_ARM_DELAY_S == pytest.approx(0.320)
