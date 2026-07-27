import time

import numpy as np
import pytest

from robots.behavior.camera_geometry import (
    CameraCorrectionProfile,
    CameraGeometryError,
    CameraIntrinsics,
    FrameCache,
    backproject_pixel_to_world,
    camera_point_from_pixel,
    fit_camera_correction_profile,
    frame_bound_hand_distance_report,
    load_camera_correction_profiles,
    project_world_to_pixel,
    robust_depth_sample,
    validated_rigid_transform,
)
from robots.behavior.toolkit import BehaviorToolResult


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


def _capture_group_frames(*, depth_value=1.0):
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0, width=5, height=5)
    return {
        camera: {
            "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
            "depth_m": np.full((5, 5), depth_value, dtype=np.float32),
            "intrinsics": intr,
            "camera_to_world": np.eye(4),
        }
        for camera in ("head", "left_wrist", "right_wrist")
    }


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


def test_pixel_projection_uses_center_depth_cluster_with_sparse_outliers():
    cache = FrameCache(ttl_s=100.0)
    depth = np.ones((9, 9), dtype=np.float32)
    depth[1, 1] = 4.0
    depth[1, 7] = 4.0
    depth[7, 1] = 4.0
    frame = cache.add(
        camera="head",
        rgb=np.zeros((9, 9, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(
            fx=4.0,
            fy=4.0,
            cx=4.0,
            cy=4.0,
            width=9,
            height=9,
        ),
        camera_to_world=np.eye(4),
        step_index=0,
    )

    result = backproject_pixel_to_world(frame, u=4, v=4, depth_window_px=7)

    assert result["depth"]["median_m"] == pytest.approx(1.0)
    assert result["depth"]["cluster_count"] == 46


def test_depth_probe_sampling_matches_projection_for_the_same_frame_pixel_and_window():
    cache = FrameCache(ttl_s=100.0)
    depth = np.full((9, 9), 0.42, dtype=np.float32)
    depth[1, 1] = 2.5
    depth[1, 7] = 2.5
    depth[7, 1] = 2.5
    frame = cache.add(
        camera="head",
        rgb=np.zeros((9, 9, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(
            fx=4.0,
            fy=4.0,
            cx=4.0,
            cy=4.0,
            width=9,
            height=9,
        ),
        camera_to_world=np.eye(4),
        step_index=0,
        frame_id="head:0:depth-probe",
    )

    sampled = robust_depth_sample(frame, u=4, v=4, window_px=7)
    projected = backproject_pixel_to_world(
        frame,
        u=4,
        v=4,
        depth_window_px=7,
    )

    assert sampled == {
        "depth_m": pytest.approx(0.42),
        "mad_m": pytest.approx(0.0),
        "valid_ratio": 1.0,
        "cluster_ratio": pytest.approx(46 / 49),
        "sample_count": 49,
        "valid_count": 49,
        "cluster_count": 46,
    }
    assert sampled["depth_m"] == pytest.approx(projected["depth"]["median_m"])
    assert sampled["mad_m"] == pytest.approx(projected["depth"]["mad_m"])
    assert sampled["valid_ratio"] == projected["depth"]["valid_ratio"]
    assert sampled["cluster_ratio"] == projected["depth"]["cluster_ratio"]


@pytest.mark.parametrize("invalid_center_depth", [0.0, np.nan])
def test_depth_probe_rejects_invalid_exact_center_with_valid_neighbors(
    invalid_center_depth,
):
    _, frame = _cache()
    frame.depth_m[:] = 0.4
    frame.depth_m[2, 2] = invalid_center_depth

    with pytest.raises(
        CameraGeometryError,
        match="selected center pixel has no finite positive metric depth",
    ):
        robust_depth_sample(frame, u=2, v=2, window_px=3)

    # The stricter exact-center rule belongs only to depth_probe sampling.
    # Keep the existing generic pixel_to_world neighborhood contract intact.
    projected = backproject_pixel_to_world(
        frame,
        u=2,
        v=2,
        depth_window_px=3,
    )
    assert projected["depth"]["median_m"] == pytest.approx(0.4)


def test_depth_probe_rejects_center_depth_from_a_different_surface_cluster():
    _, frame = _cache()
    frame.depth_m[:] = 0.4
    frame.depth_m[2, 2] = 1.2

    with pytest.raises(
        CameraGeometryError,
        match="selected center pixel does not belong to the stable depth cluster",
    ):
        robust_depth_sample(frame, u=2, v=2, window_px=3)

    projected = backproject_pixel_to_world(
        frame,
        u=2,
        v=2,
        depth_window_px=3,
    )
    assert projected["depth"]["median_m"] == pytest.approx(0.4)


def test_depth_probe_uses_neighbor_median_when_valid_center_belongs_to_cluster():
    _, frame = _cache()
    frame.depth_m[1:4, 1:4] = np.array(
        [
            [0.398, 0.401, 0.400],
            [0.399, 0.404, 0.402],
            [0.400, 0.397, 0.401],
        ],
        dtype=np.float64,
    )

    sampled = robust_depth_sample(frame, u=2, v=2, window_px=3)

    assert frame.depth_m[2, 2] == pytest.approx(0.404)
    assert sampled["depth_m"] == pytest.approx(0.400)
    assert sampled["cluster_count"] == 9


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda depth: depth.__setitem__((slice(None), slice(None)), 0.0),
            "selected center pixel has no finite positive metric depth",
        ),
        (
            lambda depth: depth.__setitem__(
                (slice(1, 4), slice(1, 4)),
                np.array(
                    [
                        [0.2, 3.0, 0.2],
                        [3.0, 0.2, 3.0],
                        [0.2, 3.0, 0.2],
                    ],
                    dtype=np.float64,
                ),
            ),
            "stable foreground cluster",
        ),
    ],
)
def test_depth_probe_sampling_fails_closed_on_unusable_depth(mutate, error):
    _, frame = _cache()
    mutate(frame.depth_m)

    with pytest.raises(CameraGeometryError, match=error):
        robust_depth_sample(frame, u=2, v=2, window_px=3)


def test_depth_probe_sampling_rejects_a_window_touching_the_image_edge():
    _, frame = _cache()

    with pytest.raises(CameraGeometryError, match="image border"):
        robust_depth_sample(frame, u=0, v=2, window_px=3)


def _rigid_transform(
    translation=(0.0, 0.0, 0.0),
    *,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    transform = np.array(
        [
            [cosine, -sine, 0.0, float(translation[0])],
            [sine, cosine, 0.0, float(translation[1])],
            [0.0, 0.0, 1.0, float(translation[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform


def _hand_reference_transforms(
    *,
    palm=(0.0, 0.0, 0.0),
    grip=(0.0, 0.0, 0.0),
    finger1=(0.0, 0.0, 0.0),
    finger2=(0.0, 0.0, 0.0),
) -> dict[str, object]:
    return {
        "palm": _rigid_transform(palm),
        "grip_point": _rigid_transform(grip),
        "finger_roots": [
            _rigid_transform(finger1),
            _rigid_transform(finger2),
        ],
    }


def _left_multiply_hand_references(
    global_transform: np.ndarray,
    references: dict[str, object],
) -> dict[str, object]:
    return {
        "palm": global_transform @ references["palm"],
        "grip_point": global_transform @ references["grip_point"],
        "finger_roots": [
            global_transform @ transform for transform in references["finger_roots"]
        ],
    }


def test_frame_bound_hand_distances_are_invariant_to_a_global_rigid_transform():
    _, frame = _cache()
    raw_target = np.array([0.13, -0.08, -0.62], dtype=np.float64)
    references = _hand_reference_transforms(
        palm=(0.03, 0.02, -0.54),
        grip=(0.04, -0.01, -0.59),
        finger1=(0.08, 0.03, -0.61),
        finger2=(0.09, -0.04, -0.60),
    )
    baseline = frame_bound_hand_distance_report(
        frame,
        raw_target_point_camera_xyz_m=raw_target,
        hand_reference_transforms_world=references,
    )

    global_transform = _rigid_transform((1.2, -0.7, 0.4), yaw_rad=0.83)
    frame.camera_to_world = global_transform @ frame.camera_to_world
    transformed = frame_bound_hand_distance_report(
        frame,
        raw_target_point_camera_xyz_m=raw_target,
        hand_reference_transforms_world=_left_multiply_hand_references(
            global_transform,
            references,
        ),
    )

    assert transformed["target_point_camera_xyz_m"] == pytest.approx(
        baseline["target_point_camera_xyz_m"]
    )
    for key in (
        "target_to_palm_m",
        "target_to_grip_point_m",
        "target_to_finger_roots_m",
        "target_to_finger_roots_individual_m",
    ):
        assert transformed[key] == pytest.approx(baseline[key])


def test_frame_bound_hand_distance_applies_camera_correction_once_and_matches_projection():
    cache = FrameCache(ttl_s=100.0)
    correction = _rigid_transform((0.04, -0.02, 0.01), yaw_rad=0.37)
    cache.set_correction_profile(
        "head",
        CameraCorrectionProfile(
            camera="head",
            raw_to_corrected_camera=correction,
            enabled=True,
            metrics={"enabled": True, "reason": "test"},
        ),
    )
    frame = cache.add(
        camera="head",
        rgb=np.zeros((7, 7, 3), dtype=np.uint8),
        depth_m=np.full((7, 7), 0.6, dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=6.0,
            fy=5.0,
            cx=3.0,
            cy=3.0,
            width=7,
            height=7,
        ),
        camera_to_world=_rigid_transform((0.8, -0.4, 0.3), yaw_rad=-0.51),
        step_index=0,
    )
    raw_target = camera_point_from_pixel(
        frame.intrinsics,
        u=4,
        v=2,
        depth_m=0.6,
    )
    effective_target = frame.correction_profile.apply_camera_point(raw_target)
    projected = backproject_pixel_to_world(
        frame,
        u=4,
        v=2,
        depth_window_px=3,
    )
    references = _hand_reference_transforms(
        palm=projected["xyz"] + np.array([0.12, 0.0, 0.0]),
        grip=projected["xyz"] + np.array([0.0, -0.08, 0.0]),
        finger1=projected["xyz"] + np.array([0.0, 0.0, 0.07]),
        finger2=projected["xyz"] + np.array([0.0, 0.0, -0.09]),
    )

    report = frame_bound_hand_distance_report(
        frame,
        raw_target_point_camera_xyz_m=raw_target,
        hand_reference_transforms_world=references,
    )

    assert report["target_point_camera_xyz_m"] == pytest.approx(effective_target)
    assert report["target_point_world_xyz_m"] == pytest.approx(projected["xyz"])
    assert report["target_to_palm_m"] == pytest.approx(0.12)
    assert report["target_to_grip_point_m"] == pytest.approx(0.08)
    assert report["target_to_finger_roots_individual_m"] == pytest.approx([0.07, 0.09])
    assert report["target_to_finger_roots_m"] == pytest.approx(0.07)


def test_frame_bound_hand_distance_uses_live_finger_root_transforms():
    _, frame = _cache()
    raw_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    closed = _hand_reference_transforms(
        palm=(0.0, 0.0, -0.9),
        grip=(0.0, 0.0, -0.95),
        finger1=(0.0, 0.03, -1.0),
        finger2=(0.0, -0.03, -1.0),
    )
    opened = _hand_reference_transforms(
        palm=(0.0, 0.0, -0.9),
        grip=(0.0, 0.0, -0.95),
        finger1=(0.0, 0.08, -1.0),
        finger2=(0.0, -0.08, -1.0),
    )

    closed_report = frame_bound_hand_distance_report(
        frame,
        raw_target_point_camera_xyz_m=raw_target,
        hand_reference_transforms_world=closed,
    )
    opened_report = frame_bound_hand_distance_report(
        frame,
        raw_target_point_camera_xyz_m=raw_target,
        hand_reference_transforms_world=opened,
    )

    assert closed_report["target_to_finger_roots_m"] == pytest.approx(0.03)
    assert opened_report["target_to_finger_roots_m"] == pytest.approx(0.08)
    assert opened_report["target_to_palm_m"] == pytest.approx(
        closed_report["target_to_palm_m"]
    )
    assert opened_report["target_to_grip_point_m"] == pytest.approx(
        closed_report["target_to_grip_point_m"]
    )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda refs: refs.pop("palm"),
            "require palm, grip_point, and finger_roots",
        ),
        (
            lambda refs: refs["finger_roots"].pop(),
            "exactly two finger roots",
        ),
        (
            lambda refs: refs["palm"].__setitem__((0, 3), np.nan),
            "NaN or infinity",
        ),
        (
            lambda refs: refs["grip_point"].__setitem__((3, 3), 2.0),
            "homogeneous last row",
        ),
        (
            lambda refs: refs["finger_roots"][0].__setitem__((0, 0), 2.0),
            "not orthonormal",
        ),
        (
            lambda refs: refs["finger_roots"][1].__setitem__((0, 0), -1.0),
            "not proper",
        ),
    ],
)
def test_frame_bound_hand_distance_rejects_incomplete_or_nonrigid_live_transforms(
    mutate,
    error,
):
    _, frame = _cache()
    references = _hand_reference_transforms()
    mutate(references)

    with pytest.raises(CameraGeometryError, match=error):
        frame_bound_hand_distance_report(
            frame,
            raw_target_point_camera_xyz_m=[0.0, 0.0, -1.0],
            hand_reference_transforms_world=references,
        )


def test_validated_rigid_transform_rejects_reflections():
    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0

    with pytest.raises(CameraGeometryError, match="not proper"):
        validated_rigid_transform(reflected, name="reflected hand link")


def test_pixel_projection_rejects_window_on_image_edge():
    _, frame = _cache()

    with pytest.raises(CameraGeometryError, match="image border"):
        backproject_pixel_to_world(frame, u=0, v=2, depth_window_px=3)


def test_observe_payload_returns_png_bytes_for_tool_result_without_pixel_dump():
    cache, _ = _cache()

    payload = cache.observe_payload("head")

    assert payload["camera"] == "head"
    assert payload["_image_bytes"].startswith(b"\x89PNG")
    assert payload["_depth_image_bytes"].startswith(b"\x89PNG")
    assert payload["image_blocks"] == ["rgb", "depth_visualization"]
    assert "rgb" not in payload
    tool_result = BehaviorToolResult(name="observe", result=payload)
    assert [block["type"] for block in tool_result.content_blocks] == [
        "text",
        "image",
        "image",
    ]
    text = tool_result.content_blocks[0]["text"]
    assert "_image_bytes" not in text
    assert "_depth_image_bytes" not in text
    assert "array(" not in text
    assert "[[[" not in text


def test_observe_never_refreshes_or_reissues_an_expired_capture():
    cache, frame = _cache()
    old_frame_id = frame.frame_id
    old_timestamp = frame.timestamp_s
    frame.timestamp_s -= cache.ttl_s + 1.0

    with pytest.raises(CameraGeometryError, match="capture expired"):
        cache.observe_payload("head")

    assert frame.frame_id == old_frame_id
    assert frame.timestamp_s == pytest.approx(old_timestamp - cache.ttl_s - 1.0)


def test_observe_reuses_same_real_capture_without_renewing_its_identity():
    cache, frame = _cache()
    timestamp = frame.timestamp_s

    first = cache.observe_payload("head")
    second = cache.observe_payload("head")

    assert first["frame_id"] == second["frame_id"] == frame.frame_id
    assert frame.timestamp_s == timestamp


def test_three_camera_capture_group_is_same_step_and_commits_atomically():
    cache = FrameCache(ttl_s=100.0)
    captured_at = time.monotonic()
    first = cache.add_capture_group(
        frames=_capture_group_frames(),
        step_index=7,
        capture_metadata={"proprio": {"values": list(range(23))}},
        timestamp_s=captured_at,
        capture_group_id="capture:7:first",
    )

    assert {frame.capture_group_id for frame in first.values()} == {"capture:7:first"}
    assert {frame.step_index for frame in first.values()} == {7}
    assert {frame.timestamp_s for frame in first.values()} == {captured_at}
    assert cache.observe_payload("left_wrist")["capture_group"] == {
        "id": "capture:7:first",
        "sim_step": 7,
        "cameras": ["head", "left_wrist", "right_wrist"],
        "age_s": pytest.approx(time.monotonic() - captured_at, abs=0.1),
    }

    malformed = _capture_group_frames(depth_value=2.0)
    malformed["right_wrist"]["depth_m"] = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(CameraGeometryError, match="depth shape"):
        cache.add_capture_group(frames=malformed, step_index=8)

    assert {
        camera: cache.latest(camera).capture_group_id
        for camera in ("head", "left_wrist", "right_wrist")
    } == {
        "head": "capture:7:first",
        "left_wrist": "capture:7:first",
        "right_wrist": "capture:7:first",
    }


def test_capture_group_requires_all_three_cameras_without_partial_publish():
    cache = FrameCache(ttl_s=100.0)
    frames = _capture_group_frames()
    del frames["right_wrist"]

    with pytest.raises(CameraGeometryError, match="missing=.*right_wrist"):
        cache.add_capture_group(frames=frames, step_index=3)

    with pytest.raises(CameraGeometryError, match="no cached"):
        cache.latest("head")


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
    np.testing.assert_allclose(
        profile.apply_camera_point([0.0, 0.0, -1.0]), [0.0, 0.0, -1.0]
    )


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
