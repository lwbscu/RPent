# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline path guard contracts and actual installed arm mesh checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.geometry import ARM_SLICES, YamCalibration, YamGeometry


class Solver:
    def solve(self, target, seed, gripper):
        return SimpleNamespace(
            success=True,
            q_target=seed + [0.08, 0, 0, 0, 0, 0],
            position_error=0.0,
            rotation_error=0.0,
        )


class RecordingGuard:
    def __init__(self, reject_interior=False):
        self.path = []
        self.reject_interior = reject_interior

    def check(self, qpos):
        self.path.append(qpos.copy())
        failed = self.reject_interior and 0.035 <= qpos[0] <= 0.065
        return {
            "ok": not failed,
            "checked": True,
            "reason": "collision_guard" if failed else None,
        }


@pytest.mark.parametrize("arm", ["left", "right"])
def test_plan_preserves_other_arm_and_both_grippers(arm):
    geometry = YamGeometry({"collision_guard": {"enabled": True}}, kinematics=Solver())
    geometry.calibration = YamCalibration({}, np.eye(4), {})
    guard = RecordingGuard()
    geometry._collision_guard = guard
    qpos = np.array([0, 0.5, 0.5, 0, 0, 0, 0.7, 0.3, 0.4, 0.6, 0, 0, 0, 0.8])
    result = geometry.plan_arm_path(arm, np.eye(4), current_qpos14=qpos)
    assert result["status"] == "Success"
    other = "right" if arm == "left" else "left"
    for sample in guard.path:
        np.testing.assert_array_equal(
            sample[ARM_SLICES[other]], qpos[ARM_SLICES[other]]
        )
        np.testing.assert_array_equal(sample[[6, 13]], qpos[[6, 13]])
    assert len(guard.path) >= 5


def test_transition_checks_interior_even_with_only_two_requested_samples():
    geometry = YamGeometry(
        {"collision_guard": {"enabled": True}, "path_joint_delta": 0.01}
    )
    guard = RecordingGuard(reject_interior=True)
    geometry._collision_guard = guard
    previous = np.zeros(14)
    target = previous.copy()
    target[0] = 0.1
    result = geometry.check_qpos_transition(previous, target, samples=2)
    assert not result["ok"]
    assert result["path_sample"] > 0
    assert guard.path[-1][0] < target[0]


def test_disabled_guard_does_not_require_mujoco_or_calibration():
    geometry = YamGeometry(kinematics=Solver())
    result = geometry.check_qpos_transition(np.zeros(14), np.zeros(14))
    assert result["ok"] and not result["checked"]


def test_enabled_guard_fails_closed_without_base_calibration():
    pytest.importorskip("mujoco")
    geometry = YamGeometry({"collision_guard": {"enabled": True}}, kinematics=Solver())
    with pytest.raises(ValueError, match="calibrated right base"):
        geometry.check_qpos_transition(np.zeros(14), np.zeros(14))


@pytest.mark.parametrize(
    "config",
    [
        {"collision_guard": {"enabled": "false"}},
        {"collision_guard": {"clearance_m": float("nan")}},
        {"path_joint_delta": 0},
        {"table_z": float("nan")},
    ],
)
def test_invalid_guard_configuration_rejected(config):
    with pytest.raises(ValueError):
        YamGeometry(config)


@pytest.fixture(scope="module")
def actual_kinematics():
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    module = pytest.importorskip("rlinf.envs.realworld.yam.kinematics")
    return module.YamKinematicsAdapter()


def actual_geometry(kinematics, *, spacing=1.2, table_z=None):
    # Synthetic geometry fixtures, never station positions or hardware commands.
    geometry = YamGeometry(
        {
            "collision_guard": {"enabled": True, "clearance_m": 0.003},
            "table_z": table_z,
        },
        kinematics=kinematics,
    )
    transform = np.eye(4)
    transform[1, 3] = spacing
    geometry.calibration = YamCalibration({}, transform, {})
    return geometry


def test_actual_mesh_separated_arms_and_gripper_exclusions(actual_kinematics):
    geometry = actual_geometry(actual_kinematics)
    qpos = np.zeros(14)
    for opening in (0.0, 1.0):
        qpos[[6, 13]] = opening
        result = geometry.check_qpos_transition(qpos, qpos)
        assert result["ok"], result
        assert result["checked"]
        assert not result["table_mesh_checked"]


def test_actual_mesh_overlapping_bases_rejected(actual_kinematics):
    geometry = actual_geometry(actual_kinematics, spacing=0)
    result = geometry.check_qpos_transition(np.zeros(14), np.zeros(14))
    assert not result["ok"]
    assert result["bodies"] == ["left_base", "right_base"]
    assert result["distance_m"] <= 0


def test_actual_mesh_self_intersection_rejected(actual_kinematics):
    geometry = actual_geometry(actual_kinematics)
    qpos = np.zeros(14)
    qpos[:6] = [
        -0.8917753895,
        1.5492289763,
        0.0889687957,
        -1.2873379707,
        0.5360336612,
        0.6165474261,
    ]
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert all(body.startswith("left_") for body in result["bodies"])
    assert result["distance_m"] < 0


def test_link_table_collision_even_when_tcp_is_above_table(actual_kinematics):
    geometry = actual_geometry(actual_kinematics, table_z=0.1)
    qpos = np.zeros(14)
    assert geometry.eef_pose("left", qpos)[2] > 0.1 + geometry.table_clearance_m
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert result["reason"] == "link_table_guard"
    assert "left_link1" in result["bodies"]


@pytest.mark.parametrize(
    "damage",
    [
        "no_geoms",
        "missing_link",
        "missing_finger_mesh",
        "wrong_joint_type",
        "extra_joint",
        "changed_exclusions",
    ],
)
def test_incomplete_or_unsupported_model_fails_closed(
    actual_kinematics, tmp_path, damage
):
    import mujoco

    xml_path = str(tmp_path / "arm.xml")
    mujoco.mj_saveLastXML(xml_path, actual_kinematics.model)
    spec = mujoco.MjSpec.from_file(xml_path)
    if damage == "no_geoms":
        for geom in list(spec.geoms):
            spec.delete(geom)
    elif damage == "missing_link":
        spec.delete(spec.body("link2").geoms[0])
    elif damage == "missing_finger_mesh":
        spec.delete(spec.body("linear_module").geoms[0])
    elif damage == "wrong_joint_type":
        spec.joint("joint1").type = mujoco.mjtJoint.mjJNT_SLIDE
    elif damage == "changed_exclusions":
        spec.delete(list(spec.excludes)[0])
    else:
        spec.body("link2").add_joint(name="extra", type=mujoco.mjtJoint.mjJNT_SLIDE)
    damaged = SimpleNamespace(model=spec.compile())
    geometry = actual_geometry(damaged, spacing=0)
    with pytest.raises(
        ValueError, match="collision_guard.*(mesh geometry|joint|exclusions)"
    ):
        geometry.check_qpos_transition(np.zeros(14), np.zeros(14))
    assert geometry._collision_guard is None


def surface_config(**overrides):
    surface = {
        "plane_z_equals_ax_by_c": [0.1, -0.05, 0.1],
        "footprint_xy": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        "depth_m": 0.3,
    }
    surface.update(overrides)
    return {
        "collision_guard": {"enabled": True, "clearance_m": 0.003},
        "table_surface": surface,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"plane_z_equals_ax_by_c": [0, float("nan"), 0]},
        {"footprint_xy": [[0, 0], [1, 0]]},
        {"footprint_xy": [[0, 0], [1, 1], [0, 1], [1, 0]]},
        {"footprint_xy": [[0, 0], [1, 0], [0.5, 0.1], [1, 1], [0, 1]]},
        {"footprint_xy": [[0, 0], [1, 0], [1, 0], [0, 1]]},
        {"depth_m": 0},
        {"depth_m": float("inf")},
        {"uncertainty_m": -0.01},
        {"typo_footprint": []},
    ],
)
def test_invalid_table_surface_rejected(overrides):
    with pytest.raises(ValueError, match="table_surface"):
        YamGeometry(surface_config(**overrides))


def test_ambiguous_or_unprotected_table_surface_rejected():
    with pytest.raises(ValueError, match="only one"):
        YamGeometry({**surface_config(), "table_z": 0.1})
    config = surface_config()
    config["collision_guard"]["enabled"] = False
    with pytest.raises(ValueError, match="requires collision_guard.enabled"):
        YamGeometry(config)


def test_surface_vertices_and_tcp_distance_use_tilt_and_finite_footprint():
    geometry = YamGeometry(surface_config(uncertainty_m=0.007))
    surface = geometry.table_surface
    assert geometry.table_guard_configured
    vertices = surface.vertices()
    np.testing.assert_allclose(
        vertices[:4, 2], 0.1 * vertices[:4, 0] - 0.05 * vertices[:4, 1] + 0.1
    )
    np.testing.assert_allclose(
        vertices[:4] - vertices[4:], np.tile([0, 0, 0.3], (4, 1))
    )
    point = np.array([0.02, 0.04, 0.2])
    expected = (0.2 - 0.1 * 0.02 + 0.05 * 0.04 - 0.1) / np.sqrt(
        1 + 0.1**2 + 0.05**2
    ) - 0.007
    assert geometry._tcp_table_clearance(point) == pytest.approx(expected)
    assert np.isinf(geometry._tcp_table_clearance(np.array([2, 0, -1])))
    assert geometry._tcp_table_clearance(np.array([0.104, 0, -0.1])) < 0
    clockwise = surface_config(footprint_xy=surface.footprint[::-1].tolist())
    assert YamGeometry(clockwise).table_guard_configured


def test_actual_finite_tilted_surface_rejects_link_when_tcp_outside(actual_kinematics):
    geometry = YamGeometry(surface_config(), kinematics=actual_kinematics)
    transform = np.eye(4)
    transform[1, 3] = 1.2
    geometry.calibration = YamCalibration({}, transform, {})
    qpos = np.zeros(14)
    assert np.isinf(geometry._tcp_table_clearance(geometry.eef_pose("left", qpos)[:3]))
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert result["reason"] == "link_table_guard"
    assert "left_link1" in result["bodies"]
    target = actual_kinematics.fk(qpos[:6], 0.0)
    plan = geometry.plan_arm_path("left", target, current_qpos14=qpos)
    assert plan["status"] == "Failure"
    assert plan["reason"] == "link_table_guard"


def test_actual_finite_surface_does_not_block_space_beyond_footprint(actual_kinematics):
    config = surface_config(
        footprint_xy=[[1.9, -0.1], [2.1, -0.1], [2.1, 0.1], [1.9, 0.1]]
    )
    geometry = YamGeometry(config, kinematics=actual_kinematics)
    transform = np.eye(4)
    transform[1, 3] = 1.2
    geometry.calibration = YamCalibration({}, transform, {})
    result = geometry.check_qpos_transition(np.zeros(14), np.zeros(14))
    assert result["ok"]
    assert result["collision_guard"]["table_mesh_checked"]


@pytest.mark.parametrize("margin", [0, -0.001, float("nan"), float("inf"), 0.011])
def test_invalid_base_link2_margin_rejected(margin):
    with pytest.raises(ValueError, match="base_link2_clearance_m"):
        YamGeometry(
            {"collision_guard": {"clearance_m": 0.01, "base_link2_clearance_m": margin}}
        )


def test_scoped_base_link2_margin_still_rejects_limit_collision(actual_kinematics):
    geometry = YamGeometry(
        {
            "collision_guard": {
                "enabled": True,
                "clearance_m": 0.01,
                "base_link2_clearance_m": 0.008,
            }
        },
        kinematics=actual_kinematics,
    )
    transform = np.eye(4)
    transform[1, 3] = 1.2
    geometry.calibration = YamCalibration({}, transform, {})
    qpos = np.zeros(14)
    assert geometry.check_qpos_transition(qpos, qpos)["ok"]
    qpos[1] = geometry.upper[0, 1]
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert result["bodies"] == ["left_base", "left_link2"]
    assert result["required_clearance_m"] == 0.008
    assert result["distance_m"] < 0.008
    qpos[0] = 1.8456845
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert result["distance_m"] < 0


def test_scoped_base_link2_margin_does_not_relax_cross_arm_pairs(actual_kinematics):
    geometry = YamGeometry(
        {
            "collision_guard": {
                "enabled": True,
                "clearance_m": 0.01,
                "base_link2_clearance_m": 0.008,
            }
        },
        kinematics=actual_kinematics,
    )
    transform = np.eye(4)
    transform[1, 3] = 0.209
    geometry.calibration = YamCalibration({}, transform, {})
    result = geometry.check_qpos_transition(np.zeros(14), np.zeros(14))
    assert not result["ok"]
    assert result["bodies"] == ["left_base", "right_base"]
    assert 0.008 < result["distance_m"] < 0.01
    assert result["required_clearance_m"] == 0.01


def test_table_uncertainty_increases_mesh_margin(actual_kinematics):
    config = surface_config(plane_z_equals_ax_by_c=[0, 0, 0.04])
    transform = np.eye(4)
    transform[1, 3] = 1.2
    geometry = YamGeometry(config, kinematics=actual_kinematics)
    geometry.calibration = YamCalibration({}, transform, {})
    qpos = np.zeros(14)
    assert geometry.check_qpos_transition(qpos, qpos)["ok"]
    config = surface_config(plane_z_equals_ax_by_c=[0, 0, 0.04], uncertainty_m=0.03)
    geometry = YamGeometry(config, kinematics=actual_kinematics)
    geometry.calibration = YamCalibration({}, transform, {})
    result = geometry.check_qpos_transition(qpos, qpos)
    assert not result["ok"]
    assert result["reason"] == "link_table_guard"
    assert result["required_clearance_m"] == pytest.approx(0.033)
    assert 0 < result["distance_m"] < 0.033
