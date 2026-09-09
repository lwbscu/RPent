# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline geometry contracts against the installed i2rt model, without hardware."""

from pathlib import Path

import numpy as np
import pytest

from robots.yam.geometry import matrix_to_xyz_wxyz, pose_to_matrix


@pytest.fixture(scope="module")
def actual_model():
    mujoco = pytest.importorskip("mujoco")
    utils = pytest.importorskip("i2rt.robots.utils")
    path = utils.combine_arm_and_gripper_xml(
        utils.ArmType.YAM, utils.GripperType.FLEXIBLE_4310
    )
    try:
        model = mujoco.MjModel.from_xml_path(path)
    finally:
        Path(path).unlink(missing_ok=True)
    return mujoco, model


@pytest.fixture(params=[np.zeros(6), np.array([0.2, 1.1, 1.4, -0.4, 0.3, 0.2])])
def actual_pose(actual_model, request):
    mujoco, model = actual_model
    data = mujoco.MjData(model)
    for index, value in enumerate(request.param, start=1):
        data.qpos[model.joint(f"joint{index}").qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_grasp_offset_and_quaternion_contract(actual_pose):
    _, model, data = actual_pose
    grasp = model.site("grasp_site").id
    tcp = model.site("tcp_site").id
    rotation = data.site_xmat[grasp].reshape(3, 3)
    np.testing.assert_allclose(rotation, data.site_xmat[tcp].reshape(3, 3))
    np.testing.assert_allclose(
        rotation.T @ (data.site_xpos[grasp] - data.site_xpos[tcp]),
        [0.0, 0.0, 0.1],
        atol=1e-10,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = data.site_xpos[grasp]
    np.testing.assert_allclose(
        pose_to_matrix(matrix_to_xyz_wxyz(transform)), transform, atol=1e-10
    )


def test_finger_slides_follow_site_y_without_moving_tcp(actual_pose):
    mujoco, model, data = actual_pose
    grasp = model.site("grasp_site").id
    rotation = data.site_xmat[grasp].reshape(3, 3).copy()
    position = data.site_xpos[grasp].copy()
    for name, expected in (("joint7", [0, 1, 0]), ("joint8", [0, -1, 0])):
        joint = model.joint(name)
        np.testing.assert_allclose(
            rotation.T @ data.xaxis[joint.id], expected, atol=1e-8
        )
        body = model.jnt_bodyid[joint.id]
        before = data.xpos[body].copy()
        data.qpos[joint.qposadr[0]] += 0.02
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            rotation.T @ (data.xpos[body] - before),
            np.asarray(expected) * 0.02,
            atol=1e-8,
        )
    np.testing.assert_allclose(data.site_xpos[grasp], position, atol=1e-10)
    np.testing.assert_allclose(
        data.site_xmat[grasp].reshape(3, 3), rotation, atol=1e-10
    )


def test_soft_tip_mesh_extends_forward_along_site_positive_z(actual_pose):
    mujoco, model, data = actual_pose
    grasp = model.site("grasp_site").id
    tcp = model.site("tcp_site").id
    rotation = data.site_xmat[grasp].reshape(3, 3)
    mount_z = (rotation.T @ (data.site_xpos[tcp] - data.site_xpos[grasp]))[2]
    tip_count = 0
    for index in range(model.ngeom):
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[index]
        if model.mesh(mesh_id).name != "soft_tips":
            continue
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
        world = vertices @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index]
        local = (world - data.site_xpos[grasp]) @ rotation
        assert local[:, 2].min() > mount_z
        assert local[:, 2].max() > 0.0
        tip_count += 1
    assert tip_count == 2
