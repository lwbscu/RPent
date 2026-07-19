import numpy as np
import pytest

from robots.behavior.camera_geometry import (
    CameraCorrectionProfile,
    CameraGeometryError,
    CameraIntrinsics,
    FrameCache,
    backproject_pixel_to_world,
    fit_camera_correction_profile,
    load_camera_correction_profiles,
    project_world_to_pixel,
)
from rpent.tools.toolkit import ToolResult


def _cache():
    cache = FrameCache(ttl_s=100.0)
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0, width=5, height=5)
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    depth = np.ones((5, 5), dtype=np.float32)
    frame = cache.add(
        camera="head",
        rgb=rgb,
        depth_m=depth,
        intrinsics=intr,
        camera_to_world=np.eye(4),
        step_index=0,
        frame_id="f0",
    )
    return cache, frame


def test_usd_minus_z_backprojection_round_trips_center_pixel():
    _, frame = _cache()

    result = backproject_pixel_to_world(frame, u=2, v=2, depth_window_px=3)

    np.testing.assert_allclose(result["xyz"], [0.0, 0.0, -1.0], atol=1e-7)
    assert result["reprojection_error_px"] <= 1e-7
    u, v, depth = project_world_to_pixel(frame, result["xyz"])
    assert (u, v, depth) == pytest.approx((2.0, 2.0, 1.0))


def test_frame_cache_rejects_stale_frame_ids():
    cache, _ = _cache()
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0, width=5, height=5)
    cache.add(
        camera="head",
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=np.ones((5, 5), dtype=np.float32),
        intrinsics=intr,
        camera_to_world=np.eye(4),
        step_index=1,
        frame_id="f1",
    )

    with pytest.raises(CameraGeometryError, match="stale frame_id"):
        cache.get_current("head", "f0")


def test_pixel_projection_rejects_missing_or_mixed_depth():
    _, frame = _cache()
    frame.depth_m[:] = 0.0

    with pytest.raises(CameraGeometryError, match="not enough valid depth"):
        backproject_pixel_to_world(frame, u=2, v=2, depth_window_px=3)

    frame.depth_m[:] = 1.0
    frame.depth_m[1:4, 1:4] = np.array(
        [
            [0.2, 3.0, 0.2],
            [3.0, 0.2, 3.0],
            [0.2, 3.0, 0.2],
        ],
        dtype=np.float64,
    )

    with pytest.raises(CameraGeometryError, match="stable foreground cluster"):
        backproject_pixel_to_world(frame, u=2, v=2, depth_window_px=3)


def test_pixel_projection_rejects_window_on_image_edge():
    _, frame = _cache()

    with pytest.raises(CameraGeometryError, match="image border"):
        backproject_pixel_to_world(frame, u=0, v=2, depth_window_px=3)


def test_observe_payload_returns_png_bytes_for_tool_result_without_pixel_dump():
    cache, _ = _cache()

    payload = cache.observe_payload("head")

    assert payload["camera"] == "head"
    assert payload["_image_bytes"].startswith(b"\x89PNG")
    assert "rgb" not in payload
    tool_result = ToolResult(name="observe", result=payload)
    assert any(block["type"] == "image" for block in tool_result.content_blocks)
    text = tool_result.content_blocks[0]["text"]
    assert "_image_bytes" not in text
    assert "array(" not in text
    assert "[[[" not in text


def _calibration_samples(offset):
    points = [
        np.array([0.0, 0.0, -1.0]),
        np.array([0.1, 0.0, -1.2]),
        np.array([0.0, 0.1, -0.8]),
        np.array([0.2, -0.1, -1.1]),
    ]
    return [
        {
            "raw_camera_xyz": point,
            "true_camera_xyz": point + np.asarray(offset, dtype=np.float64),
        }
        for point in points
    ]


def test_camera_correction_profile_accepts_good_heldout_marker_fit():
    train = _calibration_samples([0.012, -0.004, 0.0])
    heldout = _calibration_samples([0.012, -0.004, 0.0])

    profile = fit_camera_correction_profile(
        camera="head",
        train_samples=train,
        heldout_samples=heldout,
    )

    assert profile.enabled is True
    assert profile.metrics["reason"] == "accepted"
    np.testing.assert_allclose(
        profile.apply_camera_point([0.0, 0.0, -1.0]),
        [0.012, -0.004, -1.0],
        atol=1e-6,
    )


def test_enabled_camera_correction_round_trips_the_source_pixel():
    profile = fit_camera_correction_profile(
        camera="head",
        train_samples=_calibration_samples([0.012, -0.004, 0.0]),
        heldout_samples=_calibration_samples([0.012, -0.004, 0.0]),
    )
    cache = FrameCache(ttl_s=100.0, correction_profiles={"head": profile})
    frame = cache.add(
        camera="head",
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=np.ones((5, 5), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=2.0,
            fy=2.0,
            cx=2.0,
            cy=2.0,
            width=5,
            height=5,
        ),
        camera_to_world=np.eye(4),
        step_index=0,
    )

    result = backproject_pixel_to_world(frame, u=2, v=2, depth_window_px=3)

    assert result["reprojection_error_px"] <= 1e-7


def test_camera_correction_profile_rejects_bad_heldout_fit():
    train = _calibration_samples([0.012, 0.0, 0.0])
    heldout = _calibration_samples([0.0, 0.0, 0.0])

    profile = fit_camera_correction_profile(
        camera="head",
        train_samples=train,
        heldout_samples=heldout,
    )

    assert profile.enabled is False
    assert profile.metrics["reason"] == "heldout_gate_failed"
    np.testing.assert_allclose(profile.apply_camera_point([0.0, 0.0, -1.0]), [0.0, 0.0, -1.0])


def test_load_camera_correction_rejects_forged_enabled_profile(tmp_path):
    profile_path = tmp_path / "camera_profile.json"
    profile_path.write_text(
        """
        {
          "profiles": {
            "head": {
              "enabled": true,
              "raw_to_corrected_camera": [
                [1, 0, 0, 0.01],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
              ],
              "metrics": {
                "before_median_m": 0.05,
                "before_p95_m": 0.06,
                "after_median_m": 0.05,
                "after_p95_m": 0.06
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    profiles = load_camera_correction_profiles(profile_path)

    assert isinstance(profiles["head"], CameraCorrectionProfile)
    assert profiles["head"].enabled is False
    assert profiles["head"].metrics["reason"] == "heldout_gate_failed_on_load"
    np.testing.assert_allclose(
        profiles["head"].apply_camera_point([0.0, 0.0, -1.0]),
        [0.0, 0.0, -1.0],
    )
