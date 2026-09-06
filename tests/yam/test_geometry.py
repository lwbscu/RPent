# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np

from robots.yam.geometry import YamGeometry


class FakeKinematics:
    def __init__(self, base_from_grasp: np.ndarray) -> None:
        self.base_from_grasp = base_from_grasp

    def fk(self, joints: np.ndarray, gripper: float) -> np.ndarray:
        del joints, gripper
        return self.base_from_grasp.copy()


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def test_geometry_uses_legacy_t_grasp_to_cam_in_forward_direction() -> None:
    base_from_grasp = _translation(0.20, -0.10, 0.30)
    grasp_from_camera = _translation(0.04, 0.05, -0.06)
    config = {
        "calibration": {
            "world_frame": "left_base",
            "cameras": {
                "top": {
                    "cam2world_cv": np.eye(4, dtype=np.float64).tolist(),
                    "intrinsic_K": np.eye(3, dtype=np.float64).tolist(),
                },
            },
            "left_wrist": {
                "T_grasp_to_cam": grasp_from_camera.tolist(),
            },
            "T_left_base_from_right_base": np.eye(4, dtype=np.float64).tolist(),
        }
    }
    geometry = YamGeometry(
        config,
        kinematics={
            "left": FakeKinematics(base_from_grasp),
            "right": FakeKinematics(np.eye(4, dtype=np.float64)),
        },
    )

    meta = geometry.camera_meta_for_qpos(
        "left",
        {"name": "left", "intrinsic_K": np.eye(3, dtype=np.float64)},
        np.zeros(14, dtype=np.float64),
    )

    expected = base_from_grasp @ grasp_from_camera
    np.testing.assert_allclose(meta["cam2world_cv"], expected, atol=1e-12)
