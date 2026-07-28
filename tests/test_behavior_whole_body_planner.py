from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.camera_geometry import FrameCache
from robots.behavior.planner_executor import (
    _MOTION_SCOPES,
    EEF_LINK_BY_HAND,
    EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD,
    EEF_TERMINAL_POSITION_TOLERANCE_M,
    TERMINAL_COMMAND_LIMIT,
    WHOLE_BODY_ACTIVE_JOINT_NAMES,
    WHOLE_BODY_DENSE_COLLISION_STEP,
    WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M,
    WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD,
    WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M,
    WHOLE_BODY_EEF_NUMERICAL_MARGIN_M,
    WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M,
    WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD,
    WHOLE_BODY_LOCKED_JOINT_NAMES,
    PlannerExecutor,
    RealCuroboBackend,
    _apply_fixed_trajectory_hold_segments,
    _apply_single_arm_isolation_mask,
    _eef_pose_path_admission_report,
    _interpolate_joint_trajectory,
    _interpolate_whole_body_execution_trajectory,
    _joint_name_layout_sha256,
    _whole_body_indices_sha256,
    _whole_body_ordered_row_indices,
)
from robots.behavior.schemas import ACTION_DIM, ENV_ACTION_SEGMENTS


def _r1pro_joint_names() -> tuple[str, ...]:
    return (
        "base_footprint_x_joint",
        "base_footprint_y_joint",
        "base_footprint_z_joint",
        "base_footprint_rx_joint",
        "base_footprint_ry_joint",
        "base_footprint_rz_joint",
        "torso_joint1",
        "torso_joint2",
        "torso_joint3",
        "torso_joint4",
        *tuple(f"left_arm_joint{i}" for i in range(1, 8)),
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        *tuple(f"right_arm_joint{i}" for i in range(1, 8)),
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    )


def _grouped_stationary_dynamics_report(
    *,
    base_translation=0.0,
    base_yaw=0.0,
    articulation=0.0,
):
    groups = {
        "base_translation": {
            "unit": "m/s",
            "joint_names": list(WHOLE_BODY_ACTIVE_JOINT_NAMES[:2]),
            "dof_indices": [0, 1],
            "maximum": float(base_translation),
        },
        "base_yaw": {
            "unit": "rad/s",
            "joint_names": [WHOLE_BODY_ACTIVE_JOINT_NAMES[2]],
            "dof_indices": [5],
            "maximum": float(base_yaw),
        },
        "articulation": {
            "unit": "rad/s",
            "joint_names": list(WHOLE_BODY_ACTIVE_JOINT_NAMES[3:]),
            "dof_indices": [
                *range(6, 17),
                *range(19, 26),
            ],
            "maximum": float(articulation),
        },
    }
    for group in groups.values():
        maximum = group.pop("maximum")
        group.update(
            {
                "available": True,
                "dof_count": len(group["joint_names"]),
                "max_actual_velocity": maximum,
                "max_actual_velocity_joint": group["joint_names"][0],
                "max_velocity_limit": 1.0,
                "max_velocity_ratio": maximum,
                "within_control_limit": True,
            }
        )
    maximum_group = max(
        groups.values(),
        key=lambda group: group["max_actual_velocity"],
    )
    return {
        "available": True,
        "ok": True,
        "max_actual_velocity": maximum_group["max_actual_velocity"],
        "max_actual_velocity_joint": maximum_group[
            "max_actual_velocity_joint"
        ],
        "max_velocity_limit": 1.0,
        "max_velocity_ratio": maximum_group["max_velocity_ratio"],
        "velocity_groups": groups,
    }


class _WholeBodyRobot:
    def __init__(self) -> None:
        names = _r1pro_joint_names()
        self.joints = {name: object() for name in names}
        self.links = {
            EEF_LINK_BY_HAND["left"]: object(),
            EEF_LINK_BY_HAND["right"]: object(),
        }
        self.eef_link_names = dict(EEF_LINK_BY_HAND)
        self.base_idx = np.arange(0, 6)
        self.base_control_idx = np.asarray([0, 1, 5])
        self.trunk_control_idx = np.arange(6, 10)
        self.arm_control_idx = {
            "left": np.arange(10, 17),
            "right": np.arange(19, 26),
        }
        self._q = np.arange(len(names), dtype=np.float32) * 0.01

    def get_joint_positions(self):
        return self._q.copy()

    def whole_body_joint_names(self, hand):
        del hand
        return _r1pro_joint_names()


def _write_default_curobo_config(path, joint_names) -> None:
    yaml = pytest.importorskip("yaml")
    payload = {
        "robot_cfg": {
            "kinematics": {
                "ee_link": EEF_LINK_BY_HAND["left"],
                "link_names": [EEF_LINK_BY_HAND["right"]],
                "lock_joints": {},
                "cspace": {
                    "joint_names": list(joint_names),
                    "retract_config": [0.0] * len(joint_names),
                    "cspace_distance_weight": [1.0] * len(joint_names),
                    "null_space_weight": [1.0] * len(joint_names),
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("hand", ["left", "right"])
def test_whole_body_scope_config_has_selected_eef_and_exact_21_active_7_locked_joints(
    tmp_path,
    hand,
):
    yaml = pytest.importorskip("yaml")
    robot = _WholeBodyRobot()
    source = tmp_path / "r1pro_description_curobo_default.yaml"
    _write_default_curobo_config(source, tuple(robot.joints))
    backend = RealCuroboBackend(None, output_dir=tmp_path / "out")
    backend._find_robot = lambda: robot
    backend._asset_curobo_dir = lambda _robot: tmp_path

    config_path = backend._whole_body_config_path(hand)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kinematics = payload["robot_cfg"]["kinematics"]
    cspace = kinematics["cspace"]
    joint_names = tuple(cspace["joint_names"])
    locked = set(kinematics["lock_joints"])
    active = set(joint_names) - locked
    assert "whole_body" in _MOTION_SCOPES
    assert kinematics["ee_link"] == EEF_LINK_BY_HAND[hand]
    assert kinematics["link_names"] == []
    assert locked == set(WHOLE_BODY_LOCKED_JOINT_NAMES)
    assert active == set(WHOLE_BODY_ACTIVE_JOINT_NAMES)
    assert len(active) == 21
    assert len(locked) == 7
    expected_retract = [
        float(robot._q[tuple(robot.joints).index(name)]) for name in joint_names
    ]
    np.testing.assert_allclose(cspace["retract_config"], expected_retract)
    np.testing.assert_allclose(
        cspace["cspace_distance_weight"], np.ones(len(joint_names))
    )
    np.testing.assert_allclose(cspace["null_space_weight"], np.ones(len(joint_names)))


def test_whole_body_joint_merge_writes_21_active_and_preserves_7_locked():
    robot = _WholeBodyRobot()
    names = tuple(robot.joints)
    planned = np.arange(len(names), dtype=np.float32) + 100.0
    path = SimpleNamespace(joint_names=list(names), position=planned.reshape(1, -1))
    lock_state = SimpleNamespace(joint_names=list(WHOLE_BODY_LOCKED_JOINT_NAMES))
    kinematics_config = SimpleNamespace(lock_jointstate=lock_state)
    generator = SimpleNamespace(
        robot_joint_names=list(names),
        mg={
            "default": SimpleNamespace(
                kinematics=SimpleNamespace(kinematics_config=kinematics_config)
            )
        },
    )
    backend = RealCuroboBackend(None)
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default")

    merged, report = backend._merge_ik_solution_into_full_q(generator, robot, path)

    name_to_index = {name: index for index, name in enumerate(names)}
    for name in WHOLE_BODY_ACTIVE_JOINT_NAMES:
        assert merged[0, name_to_index[name]] == pytest.approx(
            planned[name_to_index[name]]
        )
    for name in WHOLE_BODY_LOCKED_JOINT_NAMES:
        assert merged[0, name_to_index[name]] == pytest.approx(
            robot._q[name_to_index[name]]
        )
    assert report["active_joint_count"] == 21
    assert report["locked_joint_count"] == 7
    assert set(report["active_joint_names"]) == set(WHOLE_BODY_ACTIVE_JOINT_NAMES)


def test_whole_body_execution_interpolation_uses_group_specific_step_caps():
    names = _r1pro_joint_names()
    q = np.zeros((2, len(names)), dtype=np.float32)
    by_name = {name: index for index, name in enumerate(names)}
    q[1, by_name["base_footprint_x_joint"]] = 0.061
    q[1, by_name["base_footprint_y_joint"]] = -0.041
    q[1, by_name["base_footprint_rz_joint"]] = np.deg2rad(3.1)
    q[1, by_name["torso_joint1"]] = 0.061
    q[1, by_name["left_arm_joint1"]] = 0.061
    q[1, by_name["right_arm_joint1"]] = -0.061

    result = _interpolate_whole_body_execution_trajectory(q, joint_names=names)
    deltas = np.abs(np.diff(result, axis=0))

    assert np.max(
        deltas[:, by_name["base_footprint_x_joint"]]
    ) <= WHOLE_BODY_DENSE_COLLISION_STEP + 1e-6
    assert np.max(
        deltas[:, by_name["base_footprint_y_joint"]]
    ) <= WHOLE_BODY_DENSE_COLLISION_STEP + 1e-6
    assert np.max(deltas[:, by_name["base_footprint_rz_joint"]]) <= (
        np.deg2rad(1.0) + 1e-6
    )
    for name in (
        "torso_joint1",
        "left_arm_joint1",
        "right_arm_joint1",
    ):
        assert np.max(
            deltas[:, by_name[name]]
        ) <= WHOLE_BODY_DENSE_COLLISION_STEP + 1e-6


def _numpy(value):
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    return np.asarray(value)


class _WholeBodyGenerator:
    batch_size = 2

    def __init__(
        self,
        *,
        solver_success: bool,
        torch_module,
        collision_waypoint: int | None = None,
    ) -> None:
        self.solver_success = bool(solver_success)
        self.collision_waypoint = collision_waypoint
        self.compute_calls: list[tuple[object, object, dict]] = []
        self.plan_overrides: list[dict[str, object]] = []
        self.collision_calls: list[tuple[np.ndarray, dict]] = []
        self.world_updates = 0
        self.path_calls = 0
        self.path = SimpleNamespace(
            joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
            position=np.stack(
                [
                    np.arange(21, dtype=np.float32) * 0.01,
                    np.arange(21, dtype=np.float32) * 0.02,
                ]
            ),
        )
        self.robot_joint_names = list(_r1pro_joint_names())
        self.rollout_retract = torch_module.full(
            (21,), -1.0, dtype=torch_module.float32
        )
        self.cspace_retract = torch_module.full((21,), -2.0, dtype=torch_module.float32)
        rollout = SimpleNamespace(
            dynamics_model=SimpleNamespace(retract_config=self.rollout_retract)
        )
        self.mg = {
            "default": SimpleNamespace(
                kinematics=SimpleNamespace(
                    joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
                    kinematics_config=SimpleNamespace(
                        cspace=SimpleNamespace(retract_config=self.cspace_retract)
                    ),
                ),
                get_all_rollout_instances=lambda: [rollout],
            )
        }

    def update_obstacles(self):
        self.world_updates += 1
        self._rpent_obstacle_refresh_metrics = {
            "clock": "time.monotonic",
            "completed_monotonic_ns": self.world_updates,
            "count": self.world_updates,
            "mode": "full" if self.world_updates == 1 else "pose_only",
            "elapsed_s": 0.001,
            "fallback": False,
        }

    def compute_trajectories(self, target_pos, target_quat, **kwargs):
        self.compute_calls.append((target_pos, target_quat, kwargs))
        self.plan_overrides.append(dict(self._rpent_plan_override))
        if self.solver_success:
            return np.asarray([True, False]), [self.path, None]
        return np.asarray([False, False]), [None, None]

    def path_to_joint_trajectory(self, path, *, get_full_js=True):
        self.path_calls += 1
        pytest.fail("whole-body path conversion must not call cuRobo get_full_js")

    def check_collisions(self, q_trajectory, **kwargs):
        q_array = _numpy(q_trajectory)
        self.collision_calls.append((q_array.copy(), kwargs))
        flags = np.zeros((len(q_array),), dtype=bool)
        if self.collision_waypoint is not None:
            flags[self.collision_waypoint] = True
        return flags


def _whole_body_plan_fixture(
    tmp_path,
    *,
    solver_success,
    collision_waypoint: int | None = None,
):
    torch = pytest.importorskip("torch")
    robot = _WholeBodyRobot()
    generator = _WholeBodyGenerator(
        solver_success=solver_success,
        torch_module=torch,
        collision_waypoint=collision_waypoint,
    )
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._torch = torch
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default")
    backend._find_robot = lambda: robot
    backend._generator = lambda **kwargs: (
        generator
        if kwargs.get("kind") == "whole_body"
        else pytest.fail(f"unexpected generator request: {kwargs!r}")
    )
    config_path = tmp_path / "whole_body.yaml"
    config_path.write_text("kind: whole_body_test\n", encoding="utf-8")
    backend._whole_body_config_path = lambda _hand: config_path
    eef_poses = {
        "left": (
            np.asarray([0.41, 0.23, 0.82]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        ),
        "right": (
            np.asarray([0.36, -0.29, 0.77]),
            np.asarray([0.1, 0.2, 0.3, 0.9])
            / np.linalg.norm(np.asarray([0.1, 0.2, 0.3, 0.9])),
        ),
    }
    backend.get_eef_pose = lambda hand: tuple(value.copy() for value in eef_poses[hand])

    def deterministic_eef_poses(_generator, q_trajectory):
        assert generator.compute_calls
        target_positions = generator.compute_calls[-1][0]
        assert len(target_positions) == 1
        selected_link, batched_target = next(iter(target_positions.items()))
        selected_hand = (
            "left"
            if selected_link == EEF_LINK_BY_HAND["left"]
            else "right"
        )
        target = _numpy(batched_target).reshape(-1, 3)[0]
        waypoint_count = len(_numpy(q_trajectory))
        positions = np.linspace(
            eef_poses[selected_hand][0],
            target,
            waypoint_count,
            dtype=np.float64,
        )
        quaternions = np.repeat(
            eef_poses[selected_hand][1].reshape(1, 4),
            waypoint_count,
            axis=0,
        )
        return positions, quaternions

    backend._curobo_eef_poses = deterministic_eef_poses
    roots = {
        "left": SimpleNamespace(prim_path="/World/held/left"),
        "right": SimpleNamespace(prim_path="/World/held/right"),
    }
    attachments = {
        hand: {EEF_LINK_BY_HAND[hand]: roots[hand]} for hand in ("left", "right")
    }
    attachment_reads: list[str] = []

    def get_attached_object(hand):
        attachment_reads.append(hand)
        return attachments[hand]

    backend.get_attached_object = get_attached_object
    return backend, generator, eef_poses, roots, attachments, attachment_reads


@pytest.mark.parametrize("hand", ["left", "right"])
def test_whole_body_plan_targets_only_selected_eef_and_merges_both_attachments(
    tmp_path,
    hand,
):
    backend, generator, eef_poses, roots, attachments, attachment_reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    inactive = "right" if hand == "left" else "left"
    requested_xyz = np.asarray([0.62, -0.17, 0.93])
    requested_quat = np.asarray([0.0, 0.4, 0.0, np.sqrt(0.84)])

    result = backend.plan_whole_body_trajectory(
        hand=hand,
        target_xyz=requested_xyz,
        target_quat_xyzw=requested_quat,
        timeout_s=4.0,
        attached_obj=attachments[hand],
    )

    assert result["ok"] is True
    assert generator.path_calls == 0
    assert len(generator.compute_calls) == 1
    assert generator.plan_overrides == [
        {
            "enable_graph": False,
            "position_only": False,
            # The planner forwards the remaining monotonic deadline budget,
            # so a few microseconds of call-entry work are expected.
            "timeout_s": pytest.approx(4.0, abs=1e-3),
        }
    ]
    target_pos, target_quat, kwargs = generator.compute_calls[0]
    assert set(target_pos) == {EEF_LINK_BY_HAND[hand]}
    assert set(target_quat) == {EEF_LINK_BY_HAND[hand]}
    selected_link = EEF_LINK_BY_HAND[hand]
    inactive_link = EEF_LINK_BY_HAND[inactive]
    np.testing.assert_allclose(
        _numpy(target_pos[selected_link]).reshape(-1, 3),
        np.broadcast_to(
            requested_xyz, _numpy(target_pos[selected_link]).reshape(-1, 3).shape
        ),
    )
    np.testing.assert_allclose(
        _numpy(target_quat[selected_link]).reshape(-1, 4),
        np.broadcast_to(
            requested_quat, _numpy(target_quat[selected_link]).reshape(-1, 4).shape
        ),
    )
    assert inactive_link not in target_pos
    assert inactive_link not in target_quat
    merged = kwargs["attached_obj"]
    assert merged[selected_link] is roots[hand]
    assert merged[inactive_link] is roots[inactive]
    assert attachment_reads == ["left", "right", "left", "right"]
    assert kwargs["ik_only"] is False
    assert kwargs["ik_world_collision_check"] is True
    assert kwargs["skip_obstacle_update"] is True
    assert kwargs["attached_obj_scale"] == {
        selected_link: 1.0,
        inactive_link: 1.0,
    }
    assert len(generator.collision_calls) == 1
    collision_q, collision_kwargs = generator.collision_calls[0]
    assert collision_q.shape[1] == 28
    live_q = _WholeBodyRobot().get_joint_positions()
    np.testing.assert_allclose(collision_q[0], live_q)
    execution_with_start = np.vstack(
        [live_q.reshape(1, -1), np.asarray(result["joint_trajectory"])]
    )
    execution_deltas = np.abs(np.diff(execution_with_start, axis=0))
    by_name = {name: index for index, name in enumerate(_r1pro_joint_names())}
    assert np.max(
        execution_deltas[:, by_name["base_footprint_x_joint"]]
    ) <= 0.02 + 1e-6
    assert np.max(
        execution_deltas[:, by_name["base_footprint_y_joint"]]
    ) <= 0.02 + 1e-6
    assert np.max(
        execution_deltas[:, by_name["base_footprint_rz_joint"]]
    ) <= np.deg2rad(1.0) + 1e-6
    for name in (
        "torso_joint1",
        "left_arm_joint1",
        "right_arm_joint1",
    ):
        assert np.max(execution_deltas[:, by_name[name]]) <= 0.02 + 1e-6
    assert collision_kwargs["self_collision_check"] is True
    assert collision_kwargs["skip_obstacle_update"] is True
    assert collision_kwargs["attached_obj"] == merged
    assert result["metrics"]["motion_scope"] == "whole_body"
    assert result["metrics"]["path_joint_merge"]["source_representation"] == (
        "active_only"
    )
    assert result["metrics"]["planning_elapsed_s"] >= 0.0
    live_by_name = dict(
        zip(_r1pro_joint_names(), _WholeBodyRobot().get_joint_positions())
    )
    expected_retract = np.asarray(
        [live_by_name[name] for name in WHOLE_BODY_ACTIVE_JOINT_NAMES],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_numpy(generator.rollout_retract), expected_retract)
    np.testing.assert_allclose(_numpy(generator.cspace_retract), expected_retract)
    assert result["metrics"]["live_retract_synchronization"] == {
        "available": True,
        "active_dof_count": 21,
        "updated_tensor_count": 2,
        "source": "call_start_robot_joint_positions",
    }
    assert result["metrics"]["collision_admission"]["admitted"] is True
    assert generator.world_updates == 1
    assert result["metrics"]["world_refresh_count"] == 1
    assert result["metrics"]["obstacle_refresh"]["mode"] == "full"
    assert result["metrics"]["obstacle_refresh"]["count"] == 1
    assert "obstacle_refresh" not in result["whole_body_certificate"]
    assert result["metrics"]["selected_eef_goal_count"] == 1
    assert result["metrics"]["inactive_eef_goal_count"] == 0
    assert result["metrics"]["eef_targets"][inactive] == {
        "role": "unconstrained",
        "target_submitted": False,
        "world_pose_gate": False,
    }
    assert np.asarray(result["joint_trajectory"]).shape[1] == 28
    assert len(result["joint_trajectory"]) >= 2


def test_short_whole_body_detour_uses_fully_recertified_direct_endpoint(
    tmp_path,
):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    start = eef_poses["left"][0]
    target = start + np.asarray([0.03, 0.0, 0.0])
    fk_calls = 0

    def detour_then_direct(_generator, q_trajectory):
        nonlocal fk_calls
        fk_calls += 1
        count = len(_numpy(q_trajectory))
        positions = np.linspace(start, target, count, dtype=np.float64)
        if fk_calls == 1:
            positions[count // 2, 1] += 0.1
        quaternions = np.repeat(
            eef_poses["left"][1].reshape(1, 4),
            count,
            axis=0,
        )
        return positions, quaternions

    backend._curobo_eef_poses = detour_then_direct
    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=eef_poses["left"][1],
        timeout_s=10.0,
    )

    assert result["ok"] is True
    audit = result["metrics"]["candidate_audit"][0]
    assert audit["solver_trajectory_eef_path"]["admitted"] is False
    assert audit["direct_endpoint_eef_path"]["admitted"] is True
    assert result["metrics"]["path_joint_merge"]["path_variant"] == (
        "call_start_to_solver_endpoint"
    )
    assert len(generator.collision_calls) == 1
    collision_q = generator.collision_calls[0][0]
    certificate = result["whole_body_certificate"]
    collision_indices = np.asarray(
        certificate["execution_collision_dense_indices"],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        result["joint_trajectory"],
        collision_q[collision_indices[1:]],
    )
    assert certificate["schema_version"] == 3
    assert certificate["collision_lineage_mode"] == (
        "source_dense_execution_subset_recertified_v1"
    )
    assert certificate["selected_eef_short_target"] is True
    assert certificate["selected_eef_path_admitted"] is True
    assert len(certificate["selected_eef_execution_positions"]) == len(
        result["joint_trajectory"]
    )
    assert len(result["joint_trajectory"]) == 7
    assert certificate["dense_collision_waypoint_count"] == 25
    assert certificate["dense_collision_waypoint_count"] > len(
        result["joint_trajectory"]
    )
    assert result["metrics"]["execution_eef_path"][
        "all_target_cartesian_step_limit_m"
    ] == pytest.approx(0.006)
    assert result["metrics"]["execution_eef_path"][
        "short_target_cartesian_step_limit_m"
    ] == pytest.approx(0.006)
    assert result["metrics"]["selected_eef_path"][
        "all_target_cartesian_step_limit_m"
    ] == pytest.approx(0.0029)
    assert result["metrics"]["selected_eef_path"][
        "max_cartesian_step_limit_m"
    ] == pytest.approx(0.0022)
    densification = audit["execution_cap_densification"]
    assert densification["refinement_count"] == 0
    assert densification["exact_path_fk_recomputed"] is False
    assert densification["exact_path_fk_reused"] is True
    assert densification["exact_path_fk_identity"] == (
        "float32-c-order-sha256+array_equal"
    )
    assert densification["collision_recertified_after_subset"] is True
    assert fk_calls == 3
    assert audit["execution_subset_attempts"][0]["collision_fk_reused"] is True
    fk_cache = result["metrics"]["selected_eef_fk_cache"]
    assert fk_cache["scope"] == "single_candidate"
    assert fk_cache["identity"] == "float32-c-order-sha256+array_equal"
    assert fk_cache["queries"] == 5
    assert fk_cache["hits"] == 2
    assert fk_cache["backend_calls"] == 3
    assert fk_cache["queries"] == fk_cache["hits"] + fk_cache["backend_calls"]
    assert fk_cache["backend_elapsed_s"] >= 0.0
    candidate_timing = audit["certificate_timing_s"]
    assert candidate_timing["clock"] == "time.monotonic"
    assert all(
        math.isfinite(value) and value >= 0.0
        for name, value in candidate_timing.items()
        if name.endswith("_s")
    )
    stage_timing = result["metrics"]["solver_stages"][0]["timing_s"]
    assert stage_timing["clock"] == "time.monotonic"
    assert stage_timing["total_s"] >= stage_timing["solver_s"]
    assert stage_timing["total_s"] >= stage_timing["certificate_s"]


def test_whole_body_near_start_solver_row_is_snapped_to_exact_live_q(tmp_path):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    generator.path.position[0] += np.float32(5e-8)
    target = eef_poses["left"][0] + np.asarray([0.03, 0.0, 0.0])

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=eef_poses["left"][1],
        timeout_s=10.0,
    )

    assert result["ok"] is True
    live_q = _WholeBodyRobot().get_joint_positions()
    collision_q = generator.collision_calls[0][0]
    np.testing.assert_array_equal(collision_q[0], live_q)
    collision_indices = np.asarray(
        result["whole_body_certificate"]["execution_collision_dense_indices"],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        result["joint_trajectory"],
        collision_q[collision_indices[1:]],
        atol=0.0,
        rtol=0.0,
    )


def test_real_backend_joint_layout_rechecks_live_robot_order(tmp_path):
    backend, _generator, _eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    robot = backend._find_robot()

    assert backend.whole_body_joint_names("left") == _r1pro_joint_names()

    robot.joints = dict(reversed(tuple(robot.joints.items())))
    with pytest.raises(RuntimeError, match="q/action joint layout diverged"):
        backend.whole_body_joint_names("left")


def test_five_degree_wrist_path_uses_geodesic_rotation_contract():
    angles = np.deg2rad(np.arange(0.0, 6.0, 1.0))
    quaternions = np.stack(
        [
            np.zeros_like(angles),
            np.zeros_like(angles),
            np.sin(angles * 0.5),
            np.cos(angles * 0.5),
        ],
        axis=1,
    )
    positions = np.zeros((len(angles), 3), dtype=np.float64)

    report = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=[0.0, 0.0, 0.0],
        target_xyz=[0.0, 0.0, 0.0],
        target_quat_xyzw=quaternions[-1],
    )

    assert report["admitted"] is True
    assert report["intentional_orientation_change"] is True
    assert report["start_orientation_error_rad"] == pytest.approx(
        np.deg2rad(5.0)
    )
    assert report["terminal_orientation_error_rad"] <= 1e-9
    assert all(report["checks"].values())


def test_whole_body_planner_admits_certified_five_degree_wrist_path(tmp_path):
    backend, _generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    start = eef_poses["left"][0]
    start_quat = eef_poses["left"][1]
    target_angle = np.deg2rad(5.0)
    target_quat = np.asarray(
        [0.0, 0.0, np.sin(target_angle * 0.5), np.cos(target_angle * 0.5)],
        dtype=np.float64,
    )

    def wrist_fk(_generator, q_trajectory):
        count = len(_numpy(q_trajectory))
        positions = np.repeat(start.reshape(1, 3), count, axis=0)
        fractions = np.linspace(0.0, 1.0, count)
        quaternions = []
        for fraction in fractions:
            quat = (1.0 - fraction) * start_quat + fraction * target_quat
            quaternions.append(quat / np.linalg.norm(quat))
        return positions, np.asarray(quaternions)

    backend._curobo_eef_poses = wrist_fk
    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=start,
        target_quat_xyzw=target_quat,
        timeout_s=10.0,
    )

    assert result["ok"] is True
    report = result["metrics"]["selected_eef_path"]
    assert report["admitted"] is True
    assert report["intentional_orientation_change"] is True
    assert all(report["checks"].values())


def test_wrist_path_that_reverses_visual_rotation_is_rejected():
    angles = np.deg2rad([0.0, 1.0, 2.0, 1.5, 2.5, 3.5, 4.5, 5.0])
    quaternions = np.stack(
        [
            np.zeros_like(angles),
            np.zeros_like(angles),
            np.sin(angles * 0.5),
            np.cos(angles * 0.5),
        ],
        axis=1,
    )
    positions = np.zeros((len(angles), 3), dtype=np.float64)

    report = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=[0.0, 0.0, 0.0],
        target_xyz=[0.0, 0.0, 0.0],
        target_quat_xyzw=quaternions[-1],
    )

    assert report["admitted"] is False
    assert report["checks"]["short_target_rotation_monotonic"] is False


def test_short_translation_rejects_tube_bounded_backtracking_loop():
    along = np.concatenate(
        [
            np.linspace(0.0, 0.03, 16),
            np.linspace(0.028, 0.0, 15),
            np.linspace(0.002, 0.03, 15),
        ]
    )
    positions = np.stack(
        [along, np.zeros_like(along), np.zeros_like(along)],
        axis=1,
    )
    quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        len(positions),
        axis=0,
    )

    report = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=[0.0, 0.0, 0.0],
        target_xyz=[0.03, 0.0, 0.0],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert report["admitted"] is False
    assert report["checks"]["short_target_axis_monotonic"] is False
    assert report["checks"]["short_target_cumulative_path"] is False
    assert report["cumulative_cartesian_path_m"] == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("cartesian_step_m", "expected_admitted"),
    [(0.0028, True), (0.0030, False)],
)
def test_long_translation_has_global_cartesian_step_certificate(
    cartesian_step_m,
    expected_admitted,
):
    waypoint_count = 21
    target_distance = cartesian_step_m * (waypoint_count - 1)
    positions = np.column_stack(
        [
            np.linspace(0.0, target_distance, waypoint_count),
            np.zeros(waypoint_count),
            np.zeros(waypoint_count),
        ]
    )
    quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        waypoint_count,
        axis=0,
    )

    report = _eef_pose_path_admission_report(
        positions,
        quaternions,
        call_start_xyz=[0.0, 0.0, 0.0],
        target_xyz=[target_distance, 0.0, 0.0],
        target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    assert report["short_target"] is False
    assert report["admitted"] is expected_admitted
    assert (
        report["checks"]["all_target_cartesian_step"]
        is expected_admitted
    )
    assert report["all_target_cartesian_step_limit_m"] == pytest.approx(
        0.0029
    )


def test_long_global_step_gate_densifies_then_rejects_before_collision(
    tmp_path,
):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    start = eef_poses["left"][0]
    target = start + np.asarray([0.08, 0.0, 0.0])

    def discontinuous_fk(_generator, q_trajectory):
        count = len(_numpy(q_trajectory))
        positions = np.repeat(start.reshape(1, 3), count, axis=0)
        positions[-1] = target
        quaternions = np.repeat(
            eef_poses["left"][1].reshape(1, 4),
            count,
            axis=0,
        )
        return positions, quaternions

    backend._curobo_eef_poses = discontinuous_fk
    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=eef_poses["left"][1],
        timeout_s=10.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "eef_path_admission_failed"
    assert generator.collision_calls == []
    reports = [
        audit["selected_eef_path"]
        for audit in result["metrics"]["candidate_audit"]
    ]
    assert reports
    for report in reports:
        assert report["checks"]["all_target_cartesian_step"] is False
        attempts = report["adaptive_densification"]["attempts"]
        assert len(attempts) == 8
        assert [row["joint_step_rad"] for row in attempts] == pytest.approx(
            [
                0.0075,
                0.00375,
                0.001875,
                0.0009375,
                0.00046875,
                0.000234375,
                0.0001171875,
                0.00005859375,
            ]
        )
        assert all(
            row["failed_checks"] == ["all_target_cartesian_step"]
            for row in attempts
        )
        assert all(row["dense_waypoints"] >= 2 for row in attempts)


def test_direct_endpoint_collision_failure_remains_zero_path(tmp_path):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(
            tmp_path,
            solver_success=True,
            collision_waypoint=0,
        )
    )
    start = eef_poses["right"][0]
    target = start + np.asarray([0.03, 0.0, 0.0])
    fk_calls = 0

    def detour_then_direct(_generator, q_trajectory):
        nonlocal fk_calls
        fk_calls += 1
        count = len(_numpy(q_trajectory))
        positions = np.linspace(start, target, count, dtype=np.float64)
        if fk_calls == 1:
            positions[count // 2, 1] += 0.1
        quaternions = np.repeat(
            eef_poses["right"][1].reshape(1, 4),
            count,
            axis=0,
        )
        return positions, quaternions

    backend._curobo_eef_poses = detour_then_direct
    result = backend.plan_whole_body_trajectory(
        hand="right",
        target_xyz=target,
        target_quat_xyzw=eef_poses["right"][1],
        timeout_s=10.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "collision_admission_failed"
    assert "joint_trajectory" not in result
    assert len(generator.collision_calls) >= 1
    assert result["metrics"]["collision_admission"][
        "collision_check_performed_candidate_count"
    ] == len(generator.collision_calls)


def test_eef_rejection_does_not_claim_collision_check(tmp_path):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    start = eef_poses["left"][0]
    target = start + np.asarray([0.03, 0.0, 0.0])

    def always_detour(_generator, q_trajectory):
        count = len(_numpy(q_trajectory))
        positions = np.linspace(start, target, count, dtype=np.float64)
        positions[count // 2, 1] += 0.1
        quaternions = np.repeat(
            eef_poses["left"][1].reshape(1, 4),
            count,
            axis=0,
        )
        return positions, quaternions

    backend._curobo_eef_poses = always_detour
    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=target,
        target_quat_xyzw=eef_poses["left"][1],
        timeout_s=10.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "eef_path_admission_failed"
    assert generator.collision_calls == []
    collision = result["metrics"]["collision_admission"]
    assert collision["post_interpolation_check"] is False
    assert collision["collision_check_performed_candidate_count"] == 0
    assert collision["colliding_waypoint_count"] is None


def test_whole_body_certifies_all_successes_and_skips_colliding_first_candidate(
    tmp_path,
):
    backend, generator, _poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    path0 = generator.path
    path1 = SimpleNamespace(
        joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
        position=np.stack(
            [
                np.arange(21, dtype=np.float32) * 0.01,
                np.arange(21, dtype=np.float32) * 0.011,
            ]
        ),
    )

    def compute(target_pos, target_quat, **kwargs):
        generator.compute_calls.append((target_pos, target_quat, kwargs))
        generator.plan_overrides.append(dict(generator._rpent_plan_override))
        return np.asarray([True, True]), [path0, path1]

    def collisions(q_trajectory, **kwargs):
        q_array = _numpy(q_trajectory)
        generator.collision_calls.append((q_array.copy(), kwargs))
        flags = np.zeros((len(q_array),), dtype=bool)
        if len(generator.collision_calls) == 1:
            flags[0] = True
        return flags

    generator.compute_trajectories = compute
    generator.check_collisions = collisions

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.asarray([0.62, 0.1, 0.85]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=10.0,
    )

    assert result["ok"] is True
    assert len(generator.compute_calls) == 1
    assert len(generator.collision_calls) == 2
    assert result["metrics"]["selected_full_trajectory_candidate"] == 1
    assert [item["certified"] for item in result["metrics"]["candidate_audit"]] == [
        False,
        True,
    ]


def test_whole_body_candidate_tie_breaks_on_shortest_full_21d_path(tmp_path):
    backend, generator, _poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    start_by_name = dict(
        zip(_r1pro_joint_names(), _WholeBodyRobot().get_joint_positions())
    )
    active_start = np.asarray(
        [start_by_name[name] for name in WHOLE_BODY_ACTIVE_JOINT_NAMES],
        dtype=np.float32,
    )
    longer_delta = np.full_like(active_start, 0.011)
    shorter_delta = np.zeros_like(active_start)
    shorter_delta[0] = 0.011
    longer = SimpleNamespace(
        joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
        position=np.stack([active_start, active_start + longer_delta]),
    )
    shorter = SimpleNamespace(
        joint_names=list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
        position=np.stack([active_start, active_start + shorter_delta]),
    )

    def compute(target_pos, target_quat, **kwargs):
        generator.compute_calls.append((target_pos, target_quat, kwargs))
        generator.plan_overrides.append(dict(generator._rpent_plan_override))
        return np.asarray([True, True]), [longer, shorter]

    generator.compute_trajectories = compute

    result = backend.plan_whole_body_trajectory(
        hand="right",
        target_xyz=np.asarray([0.55, -0.1, 0.9]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=10.0,
    )

    assert result["ok"] is True
    audit = result["metrics"]["candidate_audit"]
    assert audit[0]["execution_waypoints"] == audit[1]["execution_waypoints"]
    assert audit[1]["full_21d_path_length"] < audit[0]["full_21d_path_length"]
    assert result["metrics"]["selected_full_trajectory_candidate"] == 1
    assert result["metrics"]["full_trajectory_selection"] == (
        "fewest_execution_waypoints_then_shortest_full_21d_path"
    )


def test_empty_selected_hand_uses_position_only_motion_gen_policy(tmp_path):
    backend, generator, eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    backend.get_attached_object = lambda _hand: None

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.asarray([0.6, 0.0, 0.8]),
        target_quat_xyzw=None,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    assert generator.plan_overrides[0]["position_only"] is True
    target_quat = generator.compute_calls[0][1]
    np.testing.assert_allclose(
        _numpy(target_quat[EEF_LINK_BY_HAND["left"]])[0],
        eef_poses["left"][1],
    )
    assert result["metrics"]["eef_targets"]["left"]["orientation_mode"] == (
        "position_only_orientation_free"
    )
    assert result["metrics"]["eef_targets"]["left"]["orientation_constrained"] is False


@pytest.mark.parametrize("representation", ["active_only", "already_augmented_full"])
def test_whole_body_path_merge_accepts_both_curobo_joint_representations(
    representation,
):
    robot = _WholeBodyRobot()
    full_names = _r1pro_joint_names()
    active_names = tuple(WHOLE_BODY_ACTIVE_JOINT_NAMES)
    path_names = full_names if representation == "already_augmented_full" else active_names
    values = np.arange(2 * len(path_names), dtype=np.float32).reshape(
        2, len(path_names)
    )
    path = SimpleNamespace(joint_names=list(path_names), position=values)
    generator = SimpleNamespace(
        robot_joint_names=list(full_names),
        mg={
            "default": SimpleNamespace(
                kinematics=SimpleNamespace(joint_names=list(active_names))
            )
        },
    )
    backend = RealCuroboBackend(None)
    backend._embodiment_cls = SimpleNamespace(DEFAULT="default")
    start_q = robot.get_joint_positions()

    merged, report = backend._whole_body_path_to_full_joint_trajectory(
        generator,
        robot,
        path,
        start_q=start_q,
    )

    full_index = {name: index for index, name in enumerate(full_names)}
    path_index = {name: index for index, name in enumerate(path_names)}
    assert merged.shape == (2, 28)
    for name in active_names:
        np.testing.assert_allclose(
            merged[:, full_index[name]],
            values[:, path_index[name]],
        )
    for name in WHOLE_BODY_LOCKED_JOINT_NAMES:
        np.testing.assert_allclose(
            merged[:, full_index[name]],
            np.asarray([start_q[full_index[name]]] * 2),
        )
    assert report["source_representation"] == representation
    assert report["locked_source_entries_ignored"] == (
        7 if representation == "already_augmented_full" else 0
    )


def test_whole_body_full_trajectory_collision_failure_returns_no_joint_path(
    tmp_path,
):
    backend, generator, _eef_poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(
            tmp_path,
            solver_success=True,
            collision_waypoint=0,
        )
    )

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.asarray([0.7, 0.1, 0.8]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=4.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "collision_admission_failed"
    assert "joint_trajectory" not in result
    assert generator.path_calls == 0
    assert len(generator.compute_calls) == 2
    assert len(generator.collision_calls) == 2
    assert [item["enable_graph"] for item in generator.plan_overrides] == [
        False,
        True,
    ]
    _positions, _quaternions, kwargs = generator.compute_calls[0]
    assert kwargs["ik_only"] is False
    assert kwargs["ik_world_collision_check"] is True
    assert kwargs["skip_obstacle_update"] is True
    assert generator.world_updates == 1
    assert all(
        call_kwargs["skip_obstacle_update"] is True
        for _positions, _quaternions, call_kwargs in generator.compute_calls
    )
    collision = result["metrics"]["collision_admission"]
    assert collision["admitted"] is False
    assert collision["world_collision_check"] is True
    assert collision["self_collision_check"] is True
    assert collision["obstacle_update"] is True
    assert collision["full_trajectory"] is True
    assert collision["post_interpolation_check"] is True
    assert collision["colliding_waypoint_count"] == 2


def test_whole_body_escalates_to_graph_after_unsafe_fast_candidate(tmp_path):
    backend, generator, _poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )

    def collisions(q_trajectory, **kwargs):
        q_array = _numpy(q_trajectory)
        generator.collision_calls.append((q_array.copy(), kwargs))
        flags = np.zeros((len(q_array),), dtype=bool)
        if len(generator.collision_calls) == 1:
            flags[0] = True
        return flags

    generator.check_collisions = collisions

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.asarray([0.7, 0.1, 0.8]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=20.0,
    )

    assert result["ok"] is True
    assert len(generator.compute_calls) == 2
    assert [item["enable_graph"] for item in generator.plan_overrides] == [
        False,
        True,
    ]
    assert result["metrics"]["selected_solver_stage"] == "graph_trajopt"
    assert [item["certified"] for item in result["metrics"]["candidate_audit"]] == [
        False,
        True,
    ]


def test_whole_body_collision_backend_error_quarantines_generator(tmp_path):
    backend, generator, _poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    quarantines = []

    def broken_collision_check(*_args, **_kwargs):
        raise RuntimeError("simulated cuRobo collision backend failure")

    generator.check_collisions = broken_collision_check
    backend._quarantine_generator = lambda **kwargs: (
        quarantines.append(kwargs) or {"generator_quarantined": True}
    )

    result = backend.plan_whole_body_trajectory(
        hand="right",
        target_xyz=np.asarray([0.7, -0.1, 0.8]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        timeout_s=10.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "planner_unavailable"
    assert len(generator.compute_calls) == 1
    assert len(quarantines) == 1
    assert quarantines[0]["kind"] == "whole_body"
    assert quarantines[0]["hand"] == "right"
    assert result["metrics"]["generator_quarantine"] == {
        "generator_quarantined": True
    }


def test_whole_body_solver_error_quarantines_generator(tmp_path):
    backend, generator, _poses, _roots, _attachments, _reads = (
        _whole_body_plan_fixture(tmp_path, solver_success=True)
    )
    quarantines = []

    def broken_compute(*_args, **_kwargs):
        raise RuntimeError("simulated cuRobo solver state failure")

    generator.compute_trajectories = broken_compute
    backend._quarantine_generator = lambda **kwargs: (
        quarantines.append(kwargs) or {"generator_quarantined": True}
    )

    result = backend.plan_whole_body_trajectory(
        hand="left",
        target_xyz=np.asarray([0.6, 0.1, 0.9]),
        target_quat_xyzw=None,
        timeout_s=10.0,
    )

    assert result["ok"] is False
    assert result["stop_reason"] == "planner_unavailable"
    assert len(quarantines) == 1
    assert quarantines[0]["kind"] == "whole_body"
    assert quarantines[0]["hand"] == "left"
    assert result["metrics"]["generator_quarantine"] == {
        "generator_quarantined": True
    }


def test_whole_body_fixed_hold_preserves_all_21_active_action_slots():
    action = np.arange(ACTION_DIM, dtype=np.float32) + 10.0
    hold = np.arange(ACTION_DIM, dtype=np.float32) * -1.0

    isolated = _apply_fixed_trajectory_hold_segments(
        action,
        hold,
        hand=None,
        motion_scope="whole_body",
    )

    for segment_name in ("base", "trunk", "left_arm", "right_arm"):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_array_equal(isolated[segment], action[segment])
    for segment_name in ("left_gripper", "right_gripper"):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_array_equal(isolated[segment], hold[segment])
    with pytest.raises(ValueError, match="must not use a single-arm isolation mask"):
        _apply_single_arm_isolation_mask(
            action,
            hold,
            hand="left",
            gripper_only=False,
            motion_scope="whole_body",
        )


class _SuccessAfterOneWholeBodyStepEnv:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self._last_info = {"done": {"success": False}}
        self._gripper_latch = {"left": -0.75, "right": 0.625}

    def chunk_step(self, actions):
        self.calls.append(np.asarray(actions).copy())
        self._last_info = {"done": {"success": True}}
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


class _WholeBodyExecutionBackend:
    def __init__(self, env, *, attachments=None) -> None:
        self.env = env
        self.capture_calls: list[tuple[object, object]] = []
        self.convert_calls: list[tuple[object, object]] = []
        self.attachments = (
            {"left": None, "right": None} if attachments is None else dict(attachments)
        )
        self.attachment_reads: list[str] = []
        self._q = np.zeros((28,), dtype=np.float32)

    def capture_trajectory_hold_reference(self, *, hand, motion_scope):
        self.capture_calls.append((hand, motion_scope))
        assert hand is None
        assert motion_scope == "whole_body"
        return {
            "hand": None,
            "motion_scope": "whole_body",
            "q_indices": [2, 3, 4, 17, 18, 26, 27],
            "q_values": np.zeros(7, dtype=np.float32),
            "gripper_commands": dict(self.env._gripper_latch),
        }

    def joint_target_to_action(self, target_q, *, hand, fixed_reference):
        self.convert_calls.append((hand, fixed_reference))
        assert hand is None
        assert fixed_reference["motion_scope"] == "whole_body"
        action = np.arange(ACTION_DIM, dtype=np.float32) + 1.0
        action[ENV_ACTION_SEGMENTS["left_gripper"]] = fixed_reference[
            "gripper_commands"
        ]["left"]
        action[ENV_ACTION_SEGMENTS["right_gripper"]] = fixed_reference[
            "gripper_commands"
        ]["right"]
        return action

    def capture_single_arm_isolation_reference(self, **_kwargs):
        raise AssertionError(
            "whole-body execution must not capture single-arm isolation"
        )

    def single_arm_isolation_report(self, **_kwargs):
        raise AssertionError("whole-body execution must not run a single-arm report")

    def get_attached_object(self, hand):
        self.attachment_reads.append(hand)
        return self.attachments[hand]

    def get_joint_positions(self):
        return self._q.copy()

    def whole_body_joint_names(self, hand):
        del hand
        return _r1pro_joint_names()

    def capture_whole_body_contact_baseline(
        self, *, expected_attachments_by_hand
    ):
        del expected_attachments_by_hand
        return {"available": True, "pairs": [], "pair_count": 0}

    def whole_body_contact_report(
        self,
        *,
        baseline,
        expected_attachments_by_hand,
        allowed_expected_contact=None,
    ):
        del baseline, expected_attachments_by_hand, allowed_expected_contact
        return {"available": True, "unexpected_contact": False}

    def dynamics_report(self):
        return _grouped_stationary_dynamics_report()


def _whole_body_certificate(
    joint_trajectory,
    *,
    start_q=None,
    hand="left",
    target_xyz=None,
    target_quat_xyzw=None,
):
    trajectory = np.ascontiguousarray(joint_trajectory, dtype=np.float32)
    if start_q is None:
        start_q = np.zeros((28,), dtype=np.float32)
    start = np.ascontiguousarray(start_q, dtype=np.float32).reshape(-1)
    target = np.ascontiguousarray(
        np.zeros(3, dtype=np.float32)
        if target_xyz is None
        else np.asarray(target_xyz, dtype=np.float32).reshape(3)
    )
    target_quat = (
        None
        if target_quat_xyzw is None
        else np.ascontiguousarray(
            target_quat_xyzw,
            dtype=np.float32,
        ).reshape(4)
    )
    execution_with_start = np.ascontiguousarray(
        np.vstack([start.reshape(1, -1), trajectory]),
        dtype=np.float32,
    )
    source_dense = execution_with_start.copy()
    source_indices = np.arange(len(execution_with_start), dtype=np.int64)
    dense = np.ascontiguousarray(
        _interpolate_joint_trajectory(
            execution_with_start,
            max_inter_dist=WHOLE_BODY_DENSE_COLLISION_STEP,
        ),
        dtype=np.float32,
    )
    collision_indices = _whole_body_ordered_row_indices(
        dense,
        execution_with_start,
    )
    start_xyz = np.zeros(3, dtype=np.float32)
    dense_positions = np.ascontiguousarray(
        np.linspace(start_xyz, target, len(dense), dtype=np.float32),
        dtype=np.float32,
    )
    execution_positions = np.ascontiguousarray(
        dense_positions[collision_indices[1:]],
        dtype=np.float32,
    )
    dense_quaternions = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        len(dense),
        axis=0,
    )
    execution_quaternions = np.ascontiguousarray(
        dense_quaternions[collision_indices[1:]],
        dtype=np.float32,
    )
    trajectory_digest = hashlib.sha256(trajectory.tobytes()).hexdigest()
    dense_digest = hashlib.sha256(dense.tobytes()).hexdigest()
    source_dense_digest = hashlib.sha256(source_dense.tobytes()).hexdigest()
    dense_positions_digest = hashlib.sha256(dense_positions.tobytes()).hexdigest()
    dense_quaternions_digest = hashlib.sha256(
        dense_quaternions.tobytes()
    ).hexdigest()
    execution_positions_digest = hashlib.sha256(
        execution_positions.tobytes()
    ).hexdigest()
    execution_quaternions_digest = hashlib.sha256(
        execution_quaternions.tobytes()
    ).hexdigest()
    return {
        "schema_version": 3,
        "collision_lineage_mode": (
            "source_dense_execution_subset_recertified_v1"
        ),
        "q_encoding": "float32-c-order",
        "trajectory_sha256": trajectory_digest,
        "execution_trajectory_sha256": trajectory_digest,
        "start_q_sha256": hashlib.sha256(start.tobytes()).hexdigest(),
        "waypoint_count": int(len(trajectory)),
        "execution_waypoint_count": int(len(trajectory)),
        "execution_includes_start": False,
        "q_dimension": int(trajectory.shape[1]),
        "joint_name_layout_sha256": _joint_name_layout_sha256(
            _r1pro_joint_names()
        ),
        "active_dof_count": 21,
        "selected_eef_goal_count": 1,
        "inactive_eef_goal_count": 0,
        "attachment_hand_count": 2,
        "world_collision_check": True,
        "self_collision_check": True,
        "post_interpolation_check": True,
        "collision_free_waypoints": int(len(dense)),
        "dense_collision_waypoint_count": int(len(dense)),
        "dense_collision_checked_waypoint_count": int(len(dense)),
        "dense_collision_includes_start": True,
        "dense_collision_trajectory_sha256": dense_digest,
        "dense_collision_trajectory": dense.tolist(),
        "collision_densification_method": "joint_linear_max_inter_dist",
        "collision_densification_joint_step_rad": (
            WHOLE_BODY_DENSE_COLLISION_STEP
        ),
        "source_dense_waypoint_count": int(len(source_dense)),
        "source_dense_includes_start": True,
        "source_dense_trajectory_sha256": source_dense_digest,
        "source_dense_trajectory": source_dense.tolist(),
        "execution_source_dense_indices": source_indices.tolist(),
        "execution_source_dense_indices_sha256": (
            _whole_body_indices_sha256(source_indices)
        ),
        "execution_source_terminal_index": int(source_indices[-1]),
        "execution_collision_dense_indices": collision_indices.tolist(),
        "execution_collision_dense_indices_sha256": (
            _whole_body_indices_sha256(collision_indices)
        ),
        "execution_collision_terminal_index": int(collision_indices[-1]),
        "selected_hand": hand,
        "selected_target_xyz_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "selected_target_quat_xyzw_sha256": (
            None
            if target_quat is None
            else hashlib.sha256(target_quat.tobytes()).hexdigest()
        ),
        "selected_eef_path_admitted": True,
        "selected_eef_short_target": bool(
            np.linalg.norm(target.astype(np.float64)) <= 0.036
        ),
        "selected_eef_positions_sha256": dense_positions_digest,
        "selected_eef_dense_positions_sha256": dense_positions_digest,
        "selected_eef_dense_quaternions_sha256": dense_quaternions_digest,
        "selected_eef_dense_positions": dense_positions.tolist(),
        "selected_eef_dense_quaternions_xyzw": dense_quaternions.tolist(),
        "selected_eef_nominal_max_target_error_m": 0.0,
        "selected_eef_execution_positions_sha256": (
            execution_positions_digest
        ),
        "selected_eef_execution_quaternions_sha256": (
            execution_quaternions_digest
        ),
        "selected_eef_execution_positions": execution_positions.tolist(),
        "selected_eef_execution_quaternions_xyzw": (
            execution_quaternions.tolist()
        ),
        "selected_eef_live_waypoint_position_tolerance_m": (
            WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M
        ),
        "selected_eef_live_waypoint_orientation_tolerance_rad": (
            WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD
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
        "execution_base_xy_step_limit_m": WHOLE_BODY_DENSE_COLLISION_STEP,
        "execution_base_yaw_step_limit_rad": np.deg2rad(1.0),
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


def test_whole_body_execution_keeps_all_active_segments_latches_and_raw_success(
    tmp_path,
):
    env = _SuccessAfterOneWholeBodyStepEnv()
    backend = _WholeBodyExecutionBackend(env)
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )
    before_latches = dict(env._gripper_latch)
    joint_trajectory = np.zeros((3, 28), dtype=np.float32)
    expected_attachments = {"left": None, "right": None}

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
        joint_trajectory=joint_trajectory,
        motion_scope="whole_body",
        expected_attachments_by_hand=expected_attachments,
        whole_body_certificate=_whole_body_certificate(joint_trajectory),
    )

    assert result["primitive_success"] is True
    assert result["task_success"] is True
    assert result["_finish"] is True
    assert result["stop_reason"] == "official_task_success"
    assert result["metrics"]["executed_waypoints"] == 1
    assert len(env.calls) == 1
    assert backend.capture_calls == [(None, "whole_body")]
    assert len(backend.convert_calls) == 1
    emitted = env.calls[0][0]
    expected = np.arange(ACTION_DIM, dtype=np.float32) + 1.0
    for segment_name in ("base", "trunk", "left_arm", "right_arm"):
        segment = ENV_ACTION_SEGMENTS[segment_name]
        np.testing.assert_array_equal(emitted[segment], expected[segment])
    assert emitted[ENV_ACTION_SEGMENTS["left_gripper"]][0] == pytest.approx(
        before_latches["left"]
    )
    assert emitted[ENV_ACTION_SEGMENTS["right_gripper"]][0] == pytest.approx(
        before_latches["right"]
    )
    assert env._gripper_latch == before_latches
    assert "single_arm_isolation" not in result["metrics"]


@pytest.mark.parametrize(
    ("group", "joint_index", "delta"),
    [
        ("base_xy", 0, 0.0201),
        ("base_yaw", 5, np.deg2rad(2.0)),
        ("trunk", 6, 0.05),
        ("left_arm", 10, 0.05),
        ("right_arm", 19, 0.05),
    ],
)
def test_whole_body_tracking_requires_base_trunk_and_both_arms(
    group,
    joint_index,
    delta,
):
    del group
    robot = _WholeBodyRobot()
    backend = RealCuroboBackend(None)
    backend._find_robot = lambda: robot
    target = robot.get_joint_positions()
    baseline = backend.joint_tracking_report(target, hand=None)
    assert baseline["available"] is True
    assert baseline["reached"] is True

    target[joint_index] += delta
    report = backend.joint_tracking_report(target, hand=None)

    assert report["available"] is True
    assert report["reached"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "execution_digest",
        "source_dense_payload",
        "collision_dense_payload",
        "collision_dense_mapping",
    ],
)
def test_whole_body_certificate_tamper_executes_zero_actions(tmp_path, tamper):
    env = _SuccessAfterOneWholeBodyStepEnv()
    backend = _WholeBodyExecutionBackend(env)
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )
    joint_trajectory = np.zeros((2, 28), dtype=np.float32)
    certificate = _whole_body_certificate(joint_trajectory)
    if tamper == "execution_digest":
        certificate["trajectory_sha256"] = "0" * 64
    elif tamper == "source_dense_payload":
        certificate["source_dense_trajectory"][1][0] = 0.125
    elif tamper == "collision_dense_payload":
        certificate["dense_collision_trajectory"][1][0] = 0.125
    else:
        certificate["execution_collision_dense_indices"] = [0, 2, 1]

    with pytest.raises(
        RuntimeError,
        match="does not match its collision certificate",
    ):
        executor._execute_actions(
            None,
            hand="left",
            target_xyz=None,
            target_quat_xyzw=None,
            position_tolerance_m=0.02,
            orientation_tolerance_rad=0.087,
            timeout_s=2.0,
            require_pose=False,
            hold_steps_required=1,
            joint_trajectory=joint_trajectory,
            motion_scope="whole_body",
            expected_attachments_by_hand={"left": None, "right": None},
            whole_body_certificate=certificate,
        )

    assert env.calls == []


class _NonStationaryWholeBodyBackend(_WholeBodyExecutionBackend):
    def dynamics_report(self):
        raise AssertionError("whole-body runtime must not read dynamics")


class _UnavailableDynamicsWholeBodyBackend(_WholeBodyExecutionBackend):
    def dynamics_report(self):
        raise AssertionError("whole-body runtime must not read dynamics")


@pytest.mark.parametrize(
    "backend_type",
    [_NonStationaryWholeBodyBackend, _UnavailableDynamicsWholeBodyBackend],
)
def test_whole_body_execution_never_reads_dynamics_reporter(
    tmp_path,
    backend_type,
):
    env = _SuccessAfterOneWholeBodyStepEnv()
    backend = backend_type(env)
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )
    joint_trajectory = np.zeros((2, 28), dtype=np.float32)

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
        joint_trajectory=joint_trajectory,
        motion_scope="whole_body",
        expected_attachments_by_hand={"left": None, "right": None},
        whole_body_certificate=_whole_body_certificate(joint_trajectory),
    )

    assert result["primitive_success"] is True
    assert result["task_success"] is True
    assert result["stop_reason"] == "official_task_success"
    assert result["metrics"]["executed_waypoints"] == 1
    assert len(env.calls) == 1
    assert len(backend.convert_calls) == 1


class _NoSuccessWholeBodyEnv(_SuccessAfterOneWholeBodyStepEnv):
    def chunk_step(self, actions):
        self.calls.append(np.asarray(actions).copy())
        self._last_info = {"done": {"success": False}}
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


class _SuccessAfterTwoWholeBodyStepsEnv(_NoSuccessWholeBodyEnv):
    def chunk_step(self, actions):
        receipt = super().chunk_step(actions)
        if len(self.calls) < 2:
            return receipt
        self._last_info = {"done": {"success": True}}
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


class _ProspectiveBoundaryWholeBodyBackend(_WholeBodyExecutionBackend):
    def __init__(self, env, *, certificate) -> None:
        super().__init__(env)
        self.certificate = certificate
        execution_positions = np.asarray(
            certificate["selected_eef_execution_positions"],
            dtype=np.float64,
        )
        next_step = float(
            np.linalg.norm(execution_positions[1] - execution_positions[0])
        )
        effective_tolerance = (
            WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M
            - next_step
            - WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
        )
        self.live_boundary_position = execution_positions[0].copy()
        self.live_boundary_position[0] += effective_tolerance

    def whole_body_eef_poses(self, *, hand, q_trajectory):
        del hand
        if len(q_trajectory) == int(
            self.certificate["execution_waypoint_count"]
        ):
            return (
                self.certificate["selected_eef_execution_positions"],
                self.certificate[
                    "selected_eef_execution_quaternions_xyzw"
                ],
            )
        return (
            self.certificate["selected_eef_dense_positions"],
            self.certificate["selected_eef_dense_quaternions_xyzw"],
        )

    def get_eef_pose(self, hand):
        del hand
        position = (
            np.zeros(3, dtype=np.float64)
            if not self.env.calls
            else self.live_boundary_position
        )
        return position.copy(), np.asarray([0.0, 0.0, 0.0, 1.0])

    def joint_tracking_report(self, *, target_q, hand):
        del target_q, hand
        return {
            "available": True,
            "reached": True,
            "max_error": 0.0,
            "max_base_xy_error_m": 0.0,
            "base_yaw_error_rad": 0.0,
            "max_articulation_error_rad": 0.0,
        }


def test_whole_body_prospective_guard_uses_exact_1_001_mm_margin(tmp_path):
    env = _SuccessAfterTwoWholeBodyStepsEnv()
    joint_trajectory = np.zeros((2, 28), dtype=np.float32)
    target_xyz = np.asarray([0.004, 0.0, 0.0], dtype=np.float32)
    target_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    certificate = _whole_body_certificate(
        joint_trajectory,
        target_xyz=target_xyz,
        target_quat_xyzw=target_quat,
    )
    backend = _ProspectiveBoundaryWholeBodyBackend(
        env,
        certificate=certificate,
    )
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )

    result = executor._execute_actions(
        None,
        hand="left",
        target_xyz=target_xyz,
        target_quat_xyzw=target_quat,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.087,
        timeout_s=2.0,
        require_pose=True,
        hold_steps_required=1,
        joint_trajectory=joint_trajectory,
        motion_scope="whole_body",
        expected_attachments_by_hand={"left": None, "right": None},
        whole_body_certificate=certificate,
    )

    assert result["task_success"] is True
    assert len(env.calls) == 2
    first_waypoint = result["trace"][0][
        "whole_body_eef_waypoint_tracking"
    ]
    assert WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M == pytest.approx(
        0.001001
    )
    assert first_waypoint["prospective_guard_margin_m"] == pytest.approx(
        0.001001
    )
    assert first_waypoint["prospective_next_waypoint_step_m"] == pytest.approx(
        0.002
    )
    assert first_waypoint["effective_position_settle_tolerance_m"] == (
        pytest.approx(0.001999)
    )
    assert first_waypoint["prospective_position_bound_m"] == pytest.approx(
        WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M
    )
    assert first_waypoint["post_step_index"] == 1


class _AttachmentChangingWholeBodyBackend(_WholeBodyExecutionBackend):
    def get_attached_object(self, hand):
        self.attachment_reads.append(hand)
        read_index = len(self.attachment_reads)
        if read_index >= 4 and hand == "right":
            return {
                EEF_LINK_BY_HAND["right"]: SimpleNamespace(
                    prim_path="/World/held/replacement"
                )
            }
        return self.attachments[hand]


def test_whole_body_attachment_change_stops_before_second_action(tmp_path):
    env = _NoSuccessWholeBodyEnv()
    expected_attachments = {
        "left": {
            EEF_LINK_BY_HAND["left"]: SimpleNamespace(prim_path="/World/held/left")
        },
        "right": {
            EEF_LINK_BY_HAND["right"]: SimpleNamespace(prim_path="/World/held/right")
        },
    }
    backend = _AttachmentChangingWholeBodyBackend(
        env,
        attachments=expected_attachments,
    )
    executor = PlannerExecutor(
        env=env,
        frame_cache=FrameCache(),
        backend=backend,
        output_dir=tmp_path,
    )
    joint_trajectory = np.zeros((3, 28), dtype=np.float32)

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
        joint_trajectory=joint_trajectory,
        motion_scope="whole_body",
        expected_attachments_by_hand=expected_attachments,
        whole_body_certificate=_whole_body_certificate(joint_trajectory),
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "attachment_identity_mismatch"
    assert result["metrics"]["executed_waypoints"] == 1
    assert result["metrics"]["whole_body_attachment"]["hand"] == "right"
    assert len(env.calls) == 1
    assert backend.attachment_reads == ["left", "right", "left", "right"]


def test_whole_body_contact_monitor_allows_baseline_and_rejects_new_pair():
    contacts = [
        SimpleNamespace(body0="/World/robot/wheel", body1="/World/floor"),
    ]
    attachment_contacts = [
        SimpleNamespace(body0="/World/held/can", body1="/World/table"),
    ]
    robot = SimpleNamespace(contact_list=lambda: list(contacts))
    held_root = SimpleNamespace(contact_list=lambda: list(attachment_contacts))
    expected = {
        "left": {EEF_LINK_BY_HAND["left"]: held_root},
        "right": None,
    }
    backend = RealCuroboBackend(None)
    backend._find_robot = lambda: robot

    baseline = backend.capture_whole_body_contact_baseline(
        expected_attachments_by_hand=expected
    )
    unchanged = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )
    contacts.clear()
    disappeared = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )
    contacts.append(
        SimpleNamespace(body0="/World/robot/wheel", body1="/World/floor")
    )
    reappeared = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )
    contacts.append(
        SimpleNamespace(body0="/World/robot/left_arm", body1="/World/floor")
    )
    changed = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )

    assert baseline["available"] is True
    assert unchanged["unexpected_contact"] is False
    assert disappeared["unexpected_contact"] is False
    assert reappeared["unexpected_contact"] is True
    assert reappeared["unexpected_pairs"] == [
        ["/World/floor", "/World/robot/wheel"]
    ]
    assert changed["unexpected_contact"] is True
    assert changed["unexpected_pairs"] == [
        ["/World/floor", "/World/robot/left_arm"],
        ["/World/floor", "/World/robot/wheel"],
    ]


def test_whole_body_contact_monitor_allows_only_r1pro_wheel_floor_churn():
    robot_root = "/World/scene_0/controllable__r1pro__robot_r1"
    floor = "/World/scene_0/floors_ulujpr_0/base_link"
    contacts = []
    robot = SimpleNamespace(contact_list=lambda: list(contacts))
    backend = RealCuroboBackend(None)
    backend._find_robot = lambda: robot
    expected = {"left": None, "right": None}

    baseline = backend.capture_whole_body_contact_baseline(
        expected_attachments_by_hand=expected
    )
    contacts.append(
        SimpleNamespace(
            body0=f"{robot_root}/wheel_motor_link3",
            body1=floor,
        )
    )
    appeared = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )
    contacts.clear()
    backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )
    contacts.append(
        SimpleNamespace(
            body0=f"{robot_root}/wheel_motor_link3",
            body1=floor,
        )
    )
    reappeared = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )

    expected_pair = sorted([f"{robot_root}/wheel_motor_link3", floor])
    assert appeared["unexpected_contact"] is False
    assert appeared["allowed_support_pairs"] == [expected_pair]
    assert reappeared["unexpected_contact"] is False
    assert reappeared["allowed_support_pairs"] == [expected_pair]
    assert reappeared["allowed_support_pair_count"] == 1
    assert reappeared["support_policy"] == (
        "r1pro_wheel_motor_link0-3_to_behavior_floors_base_link"
    )


def test_real_probe_shape_two_support_pairs_then_third_wheel_is_not_collision():
    robot_root = "/World/scene_0/controllable__r1pro__robot_r1"
    floor = "/World/scene_0/floors_ulujpr_0/base_link"
    contacts = [
        SimpleNamespace(
            body0=f"{robot_root}/wheel_motor_link{index}",
            body1=floor,
        )
        for index in (0, 1)
    ]
    robot = SimpleNamespace(contact_list=lambda: list(contacts))
    backend = RealCuroboBackend(None)
    backend._find_robot = lambda: robot
    expected = {"left": None, "right": None}
    baseline = backend.capture_whole_body_contact_baseline(
        expected_attachments_by_hand=expected
    )
    contacts.append(
        SimpleNamespace(
            body0=f"{robot_root}/wheel_motor_link3",
            body1=floor,
        )
    )

    report = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )

    assert report["unexpected_contact"] is False
    assert report["current_pair_count"] == 3
    assert report["original_baseline_pair_count"] == 2
    assert report["allowed_support_pair_count"] == 3
    assert report["monitored_current_pairs"] == []


@pytest.mark.parametrize(
    ("robot_link", "scene_link"),
    [
        (
            "left_arm_link3",
            "/World/scene_0/floors_ulujpr_0/base_link",
        ),
        (
            "wheel_motor_link3",
            "/World/scene_0/walls_kitchen_0/base_link",
        ),
        (
            "wheel_motor_link4",
            "/World/scene_0/floors_ulujpr_0/base_link",
        ),
    ],
)
def test_whole_body_contact_monitor_keeps_non_support_contacts_fail_closed(
    robot_link,
    scene_link,
):
    robot_root = "/World/scene_0/controllable__r1pro__robot_r1"
    contacts = [
        SimpleNamespace(
            body0=f"{robot_root}/{robot_link}",
            body1=scene_link,
        )
    ]
    robot = SimpleNamespace(contact_list=lambda: list(contacts))
    backend = RealCuroboBackend(None)
    backend._find_robot = lambda: robot
    expected = {"left": None, "right": None}
    baseline = {
        "available": True,
        "pairs": [],
        "continuous_pairs": [],
    }

    report = backend.whole_body_contact_report(
        baseline=baseline,
        expected_attachments_by_hand=expected,
    )

    assert report["unexpected_contact"] is True
    assert report["allowed_support_pairs"] == []
    assert report["unexpected_pairs"] == [
        sorted([f"{robot_root}/{robot_link}", scene_link])
    ]
