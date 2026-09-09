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
