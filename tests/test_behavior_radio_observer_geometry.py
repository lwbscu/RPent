from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "behavior_radio_observer_geometry.py"
SPEC = importlib.util.spec_from_file_location(
    "behavior_radio_observer_geometry", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
observer_geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer_geometry)


def test_official_r1pro_eef_to_wrist_camera_transform() -> None:
    transform = observer_geometry.r1pro_eef_to_wrist_camera_transform()
    assert np.allclose(
        transform[:3, 3],
        [-0.05051, 0.0028934, -0.0651317],
        atol=1e-7,
    )
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-7)


def test_opposite_wrist_pose_uses_kit_minus_z_and_reconstructs_camera() -> None:
    result = observer_geometry.opposite_wrist_observer_pose(
        [3.3796840954584417, 4.723824129405307, 0.6571505604341787]
    )
    assert result["source"].startswith("fresh_public_pixel_to_world")
    assert np.allclose(
        result["camera_position_world_xyz"],
        [3.3796840954584417, 4.273824129405307, 0.5771505604341787],
        atol=1e-9,
    )
    assert np.allclose(
        result["eef_position_world_xyz"],
        [3.382342058676513, 4.3533191465976975, 0.5553503842522552],
        atol=1e-7,
    )
    assert np.isclose(result["optical_alignment"], 1.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(result["eef_quat_xyzw"]), 1.0, atol=1e-9)


def test_observer_pose_rejects_degenerate_or_nonfinite_geometry() -> None:
    with np.testing.assert_raises_regex(ValueError, "must differ"):
        observer_geometry.opposite_wrist_observer_pose(
            [0.0, 0.0, 0.0],
            look_offset_world_m=[0.0, 0.0, 0.0],
            camera_offset_world_m=[0.0, 0.0, 0.0],
        )
    with np.testing.assert_raises_regex(ValueError, "finite"):
        observer_geometry.opposite_wrist_observer_pose([0.0, float("nan"), 0.0])
