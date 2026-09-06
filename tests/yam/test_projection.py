# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest

import robots.yam.projection as projection
from robots.yam.projection import world_from_depth


def _camera_meta(
    *,
    width: int,
    height: int,
    intrinsic: np.ndarray,
    transform: np.ndarray | None = None,
    distortion_model: str = "none",
    distortion_coeffs: list[float] | None = None,
    projection_valid: bool = True,
    projection_limitation: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "width": width,
        "height": height,
        "intrinsic_K": intrinsic,
        "cam2world_cv": np.eye(4, dtype=np.float64) if transform is None else transform,
        "distortion_model": distortion_model,
        "distortion_coeffs": [0.0] * 5
        if distortion_coeffs is None
        else distortion_coeffs,
        "projection_valid": projection_valid,
    }
    if projection_limitation is not None:
        meta["projection_limitation"] = projection_limitation
    return meta


def test_world_from_depth_pinhole_scales_each_pixel_by_metric_depth(
    monkeypatch,
) -> None:
    projection._rays.cache_clear()
    original_import = builtins.__import__

    def fail_if_realsense_imported(name, *args, **kwargs):
        if name == "pyrealsense2":
            raise AssertionError("pinhole projection must not import pyrealsense2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_if_realsense_imported)
    depth = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    intrinsic = np.array([[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]])

    world = world_from_depth(
        depth,
        _camera_meta(width=2, height=2, intrinsic=intrinsic),
    )

    expected = np.array(
        [
            [[-0.25, -0.125, 1.0], [0.5, -0.25, 2.0]],
            [[-0.75, 0.375, 3.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(world[:1], expected[:1])
    np.testing.assert_allclose(world[1, 0], expected[1, 0])
    assert np.isnan(world[1, 1]).all()


def test_world_from_depth_uses_cv_axes_x_right_y_down() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsic = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    world = world_from_depth(
        depth,
        _camera_meta(width=2, height=2, intrinsic=intrinsic),
    )

    np.testing.assert_allclose(world[0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(world[0, 1], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(world[1, 0], [0.0, 1.0, 1.0])


def test_world_from_depth_applies_cam2world_cv_transform() -> None:
    depth = np.array([[2.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform[:3, 3] = [10.0, 20.0, 30.0]

    world = world_from_depth(
        depth,
        _camera_meta(width=1, height=1, intrinsic=intrinsic, transform=transform),
    )

    np.testing.assert_allclose(world[0, 0], [10.0, 20.0, 32.0])


@pytest.mark.parametrize(
    "meta_update, depth, match",
    [
        ({}, np.ones((1, 2), dtype=np.float32), "depth shape"),
        (
            {"intrinsic_K": np.ones((2, 2), dtype=np.float64)},
            np.ones((2, 2), dtype=np.float32),
            "intrinsic_K",
        ),
        (
            {"intrinsic_K": np.diag([-1.0, 1.0, 1.0])},
            np.ones((2, 2), dtype=np.float32),
            "positive focal",
        ),
        (
            {"cam2world_cv": np.ones((3, 3), dtype=np.float64)},
            np.ones((2, 2), dtype=np.float32),
            "cam2world_cv",
        ),
        (
            {"distortion_coeffs": [0.0, 0.0]},
            np.ones((2, 2), dtype=np.float32),
            "distortion_coeffs",
        ),
    ],
)
def test_world_from_depth_rejects_invalid_shapes_and_intrinsics(
    meta_update: dict[str, Any],
    depth: np.ndarray,
    match: str,
) -> None:
    meta = _camera_meta(width=2, height=2, intrinsic=np.eye(3, dtype=np.float64))
    meta.update(meta_update)

    with pytest.raises(ValueError, match=match):
        world_from_depth(depth, meta)


def test_world_from_depth_rejects_projection_invalid_camera_meta() -> None:
    with pytest.raises(ValueError, match="cached RGBD frame is stale"):
        world_from_depth(
            np.ones((2, 2), dtype=np.float32),
            _camera_meta(
                width=2,
                height=2,
                intrinsic=np.eye(3, dtype=np.float64),
                projection_valid=False,
                projection_limitation="cached RGBD frame is stale",
            ),
        )


def test_world_from_depth_rejects_modified_brown_conrady_deprojection() -> None:
    with pytest.raises(ValueError, match="modified_brown_conrady"):
        world_from_depth(
            np.ones((2, 2), dtype=np.float32),
            _camera_meta(
                width=2,
                height=2,
                intrinsic=np.eye(3, dtype=np.float64),
                distortion_model="modified_brown_conrady",
                distortion_coeffs=[0.01, -0.005, 0.001, -0.0005, 0.0001],
            ),
        )


@pytest.mark.parametrize(
    "distortion_model, coeffs",
    [
        ("inverse_brown_conrady", [0.01, -0.005, 0.001, -0.0005, 0.0001]),
        ("brown_conrady", [0.01, -0.005, 0.001, -0.0005, 0.0001]),
        ("ftheta", [0.01, -0.005, 0.001, -0.0005, 0.0001]),
        ("kannala_brandt4", [0.01, -0.005, 0.001, -0.0005, 0.0001]),
    ],
)
def test_world_from_depth_matches_realsense_sdk_for_distorted_representative_pixels(
    distortion_model: str,
    coeffs: list[float],
) -> None:
    rs = pytest.importorskip("pyrealsense2")
    depth = np.array(
        [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [4.5, 5.0, 5.5, 6.0]],
        dtype=np.float32,
    )
    intrinsic = np.array(
        [[100.0, 0.0, 1.5], [0.0, 120.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    meta = _camera_meta(
        width=4,
        height=3,
        intrinsic=intrinsic,
        distortion_model=distortion_model,
        distortion_coeffs=coeffs,
    )

    world = world_from_depth(depth, meta)

    intr = rs.intrinsics()
    intr.width, intr.height = 4, 3
    intr.fx, intr.fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    intr.ppx, intr.ppy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    intr.model = getattr(rs.distortion, distortion_model)
    intr.coeffs = coeffs
    for row, col in [(0, 0), (1, 2), (2, 3)]:
        expected = np.asarray(
            rs.rs2_deproject_pixel_to_point(intr, [col, row], float(depth[row, col])),
            dtype=np.float32,
        )
        np.testing.assert_allclose(world[row, col], expected, rtol=1e-6, atol=1e-6)
