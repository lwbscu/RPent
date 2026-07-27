"""RGB-D frame cache and camera geometry for BEHAVIOR planner tools."""

from __future__ import annotations

import math
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


class CameraGeometryError(ValueError):
    """Raised when a cached RGB-D frame cannot support a projection request."""


CAMERA_ALIASES: dict[str, str] = {
    "head": "head",
    "main": "head",
    "agentview": "head",
    "left": "left_wrist",
    "left_wrist": "left_wrist",
    "right": "right_wrist",
    "right_wrist": "right_wrist",
}

# Reviewed from the installed official R1Pro USD fixed joints. Matrices use the
# convention ``T_parent_from_child`` and are identical for left and right
# wrists. They include the complete Kit camera rotation, not only the camera
# origin translation documented in the compact robot-size prior.
_R1PRO_PALM_FROM_WRIST_CAMERA = np.array(
    [
        [-0.000005366182582, 0.906315917838610, 0.422600824707622, 0.0505100],
        [-0.999993452647063, -0.001534108272274, 0.003277373157896, 0.0028934],
        [0.003618650882752, -0.422598040203889, 0.906309992100728, 0.0051317],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_R1PRO_GRIP_FROM_WRIST_CAMERA = np.array(
    [
        [0.000005366182582, -0.906315917838610, -0.422600824707622, -0.0505100],
        [-0.999993452647063, -0.001534108272274, 0.003277373157896, 0.0028934],
        [-0.003618650882752, 0.422598040203889, -0.906309992100728, -0.0651317],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def canonical_camera(camera: str) -> str:
    """Return the canonical BEHAVIOR camera name."""
    key = str(camera).strip().lower()
    if key not in CAMERA_ALIASES:
        raise CameraGeometryError(
            f"unknown camera {camera!r}; expected head, left_wrist, or right_wrist"
        )
    return CAMERA_ALIASES[key]


def _as_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value, dtype=dtype)


def _homogeneous(transform: Any) -> np.ndarray:
    matrix = _as_array(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise CameraGeometryError(f"camera transform must be 4x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise CameraGeometryError("camera transform contains NaN or infinity")
    return matrix


def _rotation_angle(matrix: np.ndarray) -> float:
    trace = float(np.trace(matrix[:3, :3]))
    cos_angle = min(1.0, max(-1.0, (trace - 1.0) * 0.5))
    return float(math.acos(cos_angle))


def _rigid_transform_from_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise CameraGeometryError("calibration points must be Nx3 source/target arrays")
    if source.shape[0] < 3:
        raise CameraGeometryError("at least 3 marker correspondences are required")
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    covariance = src_centered.T @ tgt_centered / source.shape[0]
    u, _, vt = np.linalg.svd(covariance)
    det = np.linalg.det(vt.T @ u.T)
    sign = np.diag([1.0, 1.0, -1.0 if det < 0 else 1.0])
    rotation = vt.T @ sign @ u.T
    translation = tgt_centroid - rotation @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _heldout_metrics_pass_gate(
    metrics: dict[str, Any],
    *,
    min_median_improvement: float = 0.20,
    max_final_median_error_m: float = 0.02,
) -> bool:
    try:
        before_median = float(metrics["before_median_m"])
        before_p95 = float(metrics["before_p95_m"])
        after_median = float(metrics["after_median_m"])
        after_p95 = float(metrics["after_p95_m"])
    except Exception:
        return False
    if (
        before_median <= 0
        or not np.isfinite([before_median, before_p95, after_median, after_p95]).all()
    ):
        return False
    return (
        after_median <= before_median * (1.0 - float(min_median_improvement))
        and after_p95 <= before_p95
        and after_median <= float(max_final_median_error_m)
        and after_p95 <= float(max_final_median_error_m)
    )


def _sample_points(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    source_keys = (
        "raw_camera_xyz",
        "observed_camera_xyz",
        "measured_camera_xyz",
        "predicted_camera_xyz",
    )
    target_keys = (
        "true_camera_xyz",
        "marker_camera_xyz",
        "target_camera_xyz",
        "ground_truth_camera_xyz",
    )
    sources = []
    targets = []
    for sample in samples:
        source = next((sample[key] for key in source_keys if key in sample), None)
        target = next((sample[key] for key in target_keys if key in sample), None)
        if source is None or target is None:
            raise CameraGeometryError(
                "calibration sample missing camera-frame source/target xyz"
            )
        src = _as_array(source, dtype=np.float64).reshape(3)
        tgt = _as_array(target, dtype=np.float64).reshape(3)
        if not np.isfinite(src).all() or not np.isfinite(tgt).all():
            raise CameraGeometryError("calibration sample contains NaN or infinity")
        sources.append(src)
        targets.append(tgt)
    return np.stack(sources, axis=0), np.stack(targets, axis=0)


@dataclass(frozen=True)
class CameraCorrectionProfile:
    """Per-camera SE(3) correction learned only from offline marker data."""

    camera: str
    raw_to_corrected_camera: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    enabled: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", canonical_camera(self.camera))
        matrix = _homogeneous(self.raw_to_corrected_camera)
        if not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-3):
            raise CameraGeometryError(
                "camera correction rotation must be a proper SE(3) rotation"
            )
        object.__setattr__(self, "raw_to_corrected_camera", matrix)

    @classmethod
    def identity(
        cls, camera: str, *, reason: str = "identity_default"
    ) -> "CameraCorrectionProfile":
        return cls(
            camera=canonical_camera(camera),
            raw_to_corrected_camera=np.eye(4, dtype=np.float64),
            enabled=False,
            metrics={"enabled": False, "reason": reason},
        )

    @classmethod
    def from_mapping(
        cls, camera: str, value: dict[str, Any]
    ) -> "CameraCorrectionProfile":
        matrix = None
        for key in ("raw_to_corrected_camera", "transform", "matrix"):
            if key in value:
                matrix = value[key]
                break
        if matrix is None:
            return cls.identity(camera, reason="missing_transform")
        metrics = dict(value.get("metrics") or {})
        requested_enabled = bool(value.get("enabled", False))
        if requested_enabled and not _heldout_metrics_pass_gate(metrics):
            metrics.update(
                {
                    "enabled": False,
                    "reason": "heldout_gate_failed_on_load",
                    "candidate_raw_to_corrected_camera": _homogeneous(matrix).tolist(),
                }
            )
            return cls.identity(
                camera, reason="heldout_gate_failed_on_load"
            ).with_metrics(metrics)
        return cls(
            camera=camera,
            raw_to_corrected_camera=_homogeneous(matrix),
            enabled=requested_enabled,
            metrics=metrics,
        )

    def apply_camera_point(self, point_camera: Any) -> np.ndarray:
        point = _as_array(point_camera, dtype=np.float64).reshape(3)
        if not self.enabled:
            return point
        corrected = self.raw_to_corrected_camera @ np.array(
            [point[0], point[1], point[2], 1.0],
            dtype=np.float64,
        )
        return corrected[:3] / corrected[3]

    def with_metrics(self, metrics: dict[str, Any]) -> "CameraCorrectionProfile":
        return CameraCorrectionProfile(
            camera=self.camera,
            raw_to_corrected_camera=self.raw_to_corrected_camera,
            enabled=self.enabled,
            metrics=dict(metrics),
        )


def evaluate_camera_correction_profile(
    profile: CameraCorrectionProfile,
    heldout_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a correction profile on offline marker held-out data."""
    source, target = _sample_points(heldout_samples)
    before = np.linalg.norm(source - target, axis=1)
    corrected = np.stack(
        [profile.apply_camera_point(point) for point in source], axis=0
    )
    after = np.linalg.norm(corrected - target, axis=1)
    return {
        "samples": int(source.shape[0]),
        "before_median_m": float(np.median(before)),
        "before_p95_m": float(np.percentile(before, 95)),
        "after_median_m": float(np.median(after)),
        "after_p95_m": float(np.percentile(after, 95)),
        "translation_norm_m": float(
            np.linalg.norm(profile.raw_to_corrected_camera[:3, 3])
        ),
        "rotation_angle_rad": _rotation_angle(profile.raw_to_corrected_camera),
    }


def fit_camera_correction_profile(
    *,
    camera: str,
    train_samples: list[dict[str, Any]],
    heldout_samples: list[dict[str, Any]],
    min_median_improvement: float = 0.20,
    max_final_median_error_m: float = 0.02,
) -> CameraCorrectionProfile:
    """Fit and gate an offline marker-only camera-frame SE(3) correction."""
    source, target = _sample_points(train_samples)
    candidate = CameraCorrectionProfile(
        camera=camera,
        raw_to_corrected_camera=_rigid_transform_from_points(source, target),
        enabled=True,
        metrics={"source": "offline_marker_fit"},
    )
    metrics = evaluate_camera_correction_profile(candidate, heldout_samples)
    accepted = _heldout_metrics_pass_gate(
        metrics,
        min_median_improvement=min_median_improvement,
        max_final_median_error_m=max_final_median_error_m,
    )
    metrics.update(
        {
            "enabled": bool(accepted),
            "reason": "accepted" if accepted else "heldout_gate_failed",
            "min_median_improvement": float(min_median_improvement),
            "max_final_median_error_m": float(max_final_median_error_m),
            "candidate_raw_to_corrected_camera": candidate.raw_to_corrected_camera.tolist(),
        }
    )
    if accepted:
        return CameraCorrectionProfile(
            camera=camera,
            raw_to_corrected_camera=candidate.raw_to_corrected_camera,
            enabled=True,
            metrics=metrics,
        )
    return CameraCorrectionProfile.identity(
        camera, reason="heldout_gate_failed"
    ).with_metrics(metrics)


def load_camera_correction_profiles(
    path: str | Path | None,
) -> dict[str, CameraCorrectionProfile]:
    """Load optional per-camera correction profiles; missing cameras stay identity."""
    profiles = {
        camera: CameraCorrectionProfile.identity(camera)
        for camera in ("head", "left_wrist", "right_wrist")
    }
    if path is None or str(path).strip() == "":
        return profiles
    import json

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    raw_profiles = payload.get("profiles", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw_profiles, dict):
        raise CameraGeometryError(
            "camera correction profile file must contain a mapping"
        )
    for camera, value in raw_profiles.items():
        cam = canonical_camera(camera)
        if not isinstance(value, dict):
            raise CameraGeometryError(
                f"camera correction profile for {cam} must be an object"
            )
        profiles[cam] = CameraCorrectionProfile.from_mapping(cam, value)
    return profiles


def correction_profile_to_json(profile: CameraCorrectionProfile) -> dict[str, Any]:
    return {
        "camera": profile.camera,
        "enabled": profile.enabled,
        "raw_to_corrected_camera": profile.raw_to_corrected_camera.tolist(),
        "metrics": profile.metrics,
    }


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics using image coordinates u=column, v=row."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_image_shape(
        cls, shape: tuple[int, ...], *, fov_deg: float = 90.0
    ) -> "CameraIntrinsics":
        if len(shape) < 2:
            raise CameraGeometryError(f"image shape must include H,W, got {shape}")
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            raise CameraGeometryError(f"invalid image shape {shape}")
        focal = 0.5 * width / math.tan(math.radians(float(fov_deg)) * 0.5)
        return cls(
            fx=focal,
            fy=focal,
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
            width=width,
            height=height,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: Any,
        *,
        width: int,
        height: int,
    ) -> "CameraIntrinsics":
        intr = _as_array(matrix, dtype=np.float64)
        if intr.shape == (4, 4):
            intr = intr[:3, :3]
        if intr.shape != (3, 3):
            raise CameraGeometryError(f"intrinsics must be 3x3, got {intr.shape}")
        return cls(
            fx=float(intr[0, 0]),
            fy=float(intr[1, 1]),
            cx=float(intr[0, 2]),
            cy=float(intr[1, 2]),
            width=int(width),
            height=int(height),
        )

    def validate(self) -> None:
        values = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.isfinite(values).all():
            raise CameraGeometryError("intrinsics contain NaN or infinity")
        if self.fx <= 0 or self.fy <= 0:
            raise CameraGeometryError("intrinsics focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise CameraGeometryError("intrinsics image dimensions must be positive")


@dataclass
class RgbdFrame:
    """One synchronous RGB-D capture and the camera pose used for projection."""

    camera: str
    frame_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    camera_to_world: np.ndarray
    step_index: int
    capture_group_id: str | None = None
    capture_metadata: dict[str, Any] = field(default_factory=dict)
    correction_profile: CameraCorrectionProfile | None = None
    timestamp_s: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.camera = canonical_camera(self.camera)
        self.rgb = _as_array(self.rgb)
        self.depth_m = _normalize_depth(self.depth_m)
        if self.rgb.ndim != 3 or self.rgb.shape[2] not in (3, 4):
            raise CameraGeometryError(
                f"rgb image must be HxWx3/4, got {self.rgb.shape}"
            )
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise CameraGeometryError(
                f"depth shape {self.depth_m.shape} does not match rgb {self.rgb.shape[:2]}"
            )
        if (
            self.intrinsics.width != self.rgb.shape[1]
            or self.intrinsics.height != self.rgb.shape[0]
        ):
            raise CameraGeometryError("intrinsics dimensions do not match rgb image")
        self.intrinsics.validate()
        self.camera_to_world = _homogeneous(self.camera_to_world)
        if self.capture_group_id is not None:
            self.capture_group_id = str(self.capture_group_id)
        self.capture_metadata = deepcopy(dict(self.capture_metadata))
        if self.correction_profile is None:
            self.correction_profile = CameraCorrectionProfile.identity(self.camera)
        elif self.correction_profile.camera != self.camera:
            raise CameraGeometryError(
                "correction profile camera does not match RGB-D frame camera"
            )


def _normalize_depth(depth: Any) -> np.ndarray:
    """Normalize one metric optical-axis depth image without guessing its unit.

    Production BEHAVIOR captures bind this array to OmniGibson's
    ``depth_linear`` / ``distance_to_image_plane`` modality, whose declared
    unit is metres.  Silently inferring millimetres from scene-dependent values
    would turn a unit or modality wiring error into plausible-but-wrong
    geometry, so this boundary only normalizes shape and numeric dtype.
    """

    array = _as_array(depth, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise CameraGeometryError(f"depth must be HxW, got {array.shape}")
    return array.astype(np.float64, copy=False)


def _png_bytes(rgb: Any) -> bytes:
    array = _as_array(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CameraGeometryError(f"rgb image must be HxWx3/4, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        finite = np.nan_to_num(
            array.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0
        )
        if finite.size and finite.max() <= 1.0:
            finite = finite * 255.0
        array = np.clip(finite, 0.0, 255.0).astype(np.uint8)
    buffer = BytesIO()
    try:
        import imageio.v2 as imageio

        imageio.imwrite(buffer, array, format="png")
    except Exception as exc:
        raise CameraGeometryError(
            f"failed to encode RGB observation as PNG: {exc}"
        ) from exc
    return buffer.getvalue()


def _depth_png_bytes(depth_m: Any) -> tuple[bytes, dict[str, Any]]:
    """Encode metric depth as a robust inverse-depth visualization.

    The visualization is intentionally lossy. Metric depth stays simulator-side
    for :func:`backproject_pixel_to_world`; the VLM receives an inspectable image
    without a large numeric pixel dump.
    """
    depth = _normalize_depth(depth_m)
    valid_mask = np.isfinite(depth) & (depth > 0)
    valid = depth[valid_mask]
    if valid.size == 0:
        raise CameraGeometryError("cannot visualize depth without valid samples")
    near_m, far_m = np.percentile(valid, [2.0, 98.0]).astype(np.float64)
    if not np.isfinite([near_m, far_m]).all():
        raise CameraGeometryError("cannot visualize non-finite depth bounds")
    if far_m <= near_m:
        half_span = max(0.001, abs(float(near_m)) * 0.025)
        near_m = max(0.0, float(near_m) - half_span)
        far_m = float(far_m) + half_span
    normalized = np.clip((depth - near_m) / (far_m - near_m), 0.0, 1.0)
    intensity = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    visual = np.repeat(intensity[..., None], 3, axis=2)
    # Magenta is reserved for missing / invalid sensor depth.
    visual[~valid_mask] = np.array([255, 0, 255], dtype=np.uint8)
    return _png_bytes(visual), {
        "format": "png",
        "mapping": "inverse_linear_percentile_2_98",
        "source_modality": "depth_linear",
        "measurement": "distance_to_image_plane",
        "unit": "m",
        "near_m": float(near_m),
        "far_m": float(far_m),
        "invalid_color_rgb": [255, 0, 255],
        "valid_ratio": float(valid.size / depth.size),
    }


def _to_int_pixel(name: str, value: Any) -> int:
    if value is None:
        raise CameraGeometryError(f"{name} is required")
    try:
        rounded = int(value)
    except Exception as exc:
        raise CameraGeometryError(f"{name} must be an integer pixel") from exc
    if float(value) != float(rounded):
        raise CameraGeometryError(f"{name} must be an integer pixel")
    return rounded


def _valid_depth_values(depth: np.ndarray) -> np.ndarray:
    return depth[np.isfinite(depth) & (depth > 0)]


def _cluster_depth(
    frame: RgbdFrame,
    *,
    u: int,
    v: int,
    depth_window_px: int,
    require_selected_pixel_depth: bool = False,
) -> dict[str, Any]:
    radius = max(0, int(depth_window_px) // 2)
    if u < 0 or v < 0 or u >= frame.intrinsics.width or v >= frame.intrinsics.height:
        raise CameraGeometryError(
            f"pixel ({u},{v}) outside image {frame.intrinsics.width}x{frame.intrinsics.height}"
        )
    if radius and (
        u - radius < 0
        or v - radius < 0
        or u + radius >= frame.intrinsics.width
        or v + radius >= frame.intrinsics.height
    ):
        raise CameraGeometryError(
            "pixel window touches image border; refusing edge projection"
        )
    selected_depth = float(frame.depth_m[v, u])
    if require_selected_pixel_depth and (
        not math.isfinite(selected_depth) or selected_depth <= 0.0
    ):
        raise CameraGeometryError(
            "selected center pixel has no finite positive metric depth"
        )
    crop = frame.depth_m[v - radius : v + radius + 1, u - radius : u + radius + 1]
    valid = _valid_depth_values(crop)
    valid_ratio = float(valid.size / crop.size) if crop.size else 0.0
    if valid.size < max(3, crop.size // 5):
        raise CameraGeometryError("not enough valid depth samples around pixel")

    center_radius = min(1, radius)
    center_crop = frame.depth_m[
        v - center_radius : v + center_radius + 1,
        u - center_radius : u + center_radius + 1,
    ]
    center_valid = _valid_depth_values(center_crop)
    if center_valid.size < max(1, center_crop.size // 3):
        raise CameraGeometryError("not enough valid center depth samples around pixel")
    center_median = float(np.median(center_valid))
    center_mad = float(np.median(np.abs(center_valid - center_median)))
    center_scale = max(
        0.002,
        0.015 * max(center_median, 1e-6),
        2.5 * 1.4826 * center_mad,
    )
    cluster = valid[np.abs(valid - center_median) <= center_scale]
    # A substantial second population means the requested window straddles an
    # object boundary. Refuse it instead of averaging foreground/background.
    if cluster.size < max(3, int(math.ceil(valid.size * 0.60))):
        raise CameraGeometryError("depth window has no stable foreground cluster")
    if require_selected_pixel_depth and (
        abs(selected_depth - center_median) > center_scale
    ):
        raise CameraGeometryError(
            "selected center pixel does not belong to the stable depth cluster"
        )
    cluster_median = float(np.median(cluster))
    cluster_mad = float(np.median(np.abs(cluster - cluster_median)))
    if cluster_median <= 0 or not math.isfinite(cluster_median):
        raise CameraGeometryError("depth cluster produced invalid depth")
    if cluster_mad / max(cluster_median, 1e-6) > 0.04:
        raise CameraGeometryError(
            "depth cluster is too dispersed for reliable projection"
        )

    return {
        "depth_m": cluster_median,
        "mad_m": cluster_mad,
        "valid_ratio": valid_ratio,
        "cluster_ratio": float(cluster.size / crop.size),
        "sample_count": int(crop.size),
        "valid_count": int(valid.size),
        "cluster_count": int(cluster.size),
    }


def robust_depth_sample(
    frame_or_depth: RgbdFrame | Any,
    *,
    u: Any,
    v: Any,
    window_px: int = 7,
) -> dict[str, Any]:
    """Return the same center-cluster median/MAD used by public projection."""
    if isinstance(frame_or_depth, RgbdFrame):
        frame = frame_or_depth
    else:
        depth = _normalize_depth(frame_or_depth)
        height, width = depth.shape
        intrinsics = CameraIntrinsics(
            fx=1.0,
            fy=1.0,
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
            width=width,
            height=height,
        )
        frame = RgbdFrame(
            camera="head",
            frame_id="robust-depth-sample",
            rgb=np.zeros((height, width, 3), dtype=np.uint8),
            depth_m=depth,
            intrinsics=intrinsics,
            camera_to_world=np.eye(4),
            step_index=0,
        )
    return _cluster_depth(
        frame,
        u=_to_int_pixel("u", u),
        v=_to_int_pixel("v", v),
        depth_window_px=int(window_px),
        require_selected_pixel_depth=True,
    )


def camera_point_from_pixel(
    intrinsics: CameraIntrinsics,
    *,
    u: int,
    v: int,
    depth_m: float,
) -> np.ndarray:
    """Back-project an image pixel into USD camera coordinates with -Z forward."""
    intrinsics.validate()
    depth = float(depth_m)
    if depth <= 0 or not math.isfinite(depth):
        raise CameraGeometryError("depth_m must be positive and finite")
    x = (float(u) - intrinsics.cx) * depth / intrinsics.fx
    y = (intrinsics.cy - float(v)) * depth / intrinsics.fy
    return np.array([x, y, -depth], dtype=np.float64)


def pixel_from_camera_point(
    intrinsics: CameraIntrinsics, point_camera: Any
) -> tuple[float, float]:
    """Project a USD camera-space point to image coordinates."""
    p = _as_array(point_camera, dtype=np.float64).reshape(3)
    depth = -float(p[2])
    if depth <= 0 or not math.isfinite(depth):
        raise CameraGeometryError("point is behind the USD camera -Z optical axis")
    u = intrinsics.fx * float(p[0]) / depth + intrinsics.cx
    v = intrinsics.cy - intrinsics.fy * float(p[1]) / depth
    return float(u), float(v)


def transform_point(transform: Any, point: Any) -> np.ndarray:
    matrix = _homogeneous(transform)
    p = _as_array(point, dtype=np.float64).reshape(3)
    out = matrix @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return out[:3] / out[3]


def validated_rigid_transform(value: Any, *, name: str) -> np.ndarray:
    """Return one finite proper SE(3) transform or fail closed.

    Runtime hand-distance receipts bind complete live link transforms.  Merely
    accepting a finite 4x4 array would also accept scale, shear, reflection, or
    a malformed homogeneous row and could make an apparently precise distance
    geometrically meaningless.
    """

    matrix = _homogeneous(value)
    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        atol=1e-7,
        rtol=0.0,
    ):
        raise CameraGeometryError(f"{name} homogeneous last row is invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    ):
        raise CameraGeometryError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or not math.isclose(
        determinant, 1.0, abs_tol=1e-5
    ):
        raise CameraGeometryError(f"{name} rotation is not proper")
    return matrix


def rigid_transform_residual(
    observed: Any,
    expected: Any,
) -> dict[str, float]:
    """Return translation and rotation residuals between two SE(3) transforms."""

    actual = validated_rigid_transform(observed, name="observed transform")
    reference = validated_rigid_transform(expected, name="expected transform")
    translation_error_m = float(np.linalg.norm(actual[:3, 3] - reference[:3, 3]))
    rotation_delta = reference[:3, :3].T @ actual[:3, :3]
    trace = float(np.trace(rotation_delta))
    cosine = min(1.0, max(-1.0, (trace - 1.0) * 0.5))
    rotation_error_rad = float(math.acos(cosine))
    return {
        "translation_error_m": translation_error_m,
        "rotation_error_rad": rotation_error_rad,
        "rotation_error_deg": math.degrees(rotation_error_rad),
    }


def r1pro_wrist_camera_reference_transforms() -> dict[str, np.ndarray]:
    """Return reviewed full R1Pro palm/grip-from-wrist-camera transforms."""

    return {
        "palm_from_camera": validated_rigid_transform(
            _R1PRO_PALM_FROM_WRIST_CAMERA.copy(),
            name="reviewed R1Pro palm-from-camera transform",
        ),
        "grip_point_from_camera": validated_rigid_transform(
            _R1PRO_GRIP_FROM_WRIST_CAMERA.copy(),
            name="reviewed R1Pro grip-from-camera transform",
        ),
    }


def frame_bound_hand_distance_report(
    frame: RgbdFrame,
    *,
    raw_target_point_camera_xyz_m: Any,
    hand_reference_transforms_world: dict[str, Any],
) -> dict[str, Any]:
    """Measure one frame-bound RGB-D point against captured R1Pro references.

    The input point is the raw USD camera point produced by pinhole
    back-projection.  The frame's reviewed correction profile is applied
    exactly as it is for :func:`backproject_pixel_to_world`, then the effective
    point is transformed to world.  Distances are computed in world against
    link transforms captured with this RGB-D group; corrected and raw camera
    coordinates are never mixed.
    """

    raw_point = _as_array(raw_target_point_camera_xyz_m, dtype=np.float64).reshape(3)
    if not np.isfinite(raw_point).all():
        raise CameraGeometryError("target camera point contains NaN or infinity")
    effective_point = frame.correction_profile.apply_camera_point(raw_point)
    if not np.isfinite(effective_point).all():
        raise CameraGeometryError(
            "corrected target camera point contains NaN or infinity"
        )
    target_world = transform_point(frame.camera_to_world, effective_point)
    if not np.isfinite(target_world).all():
        raise CameraGeometryError("target world point contains NaN or infinity")

    if not isinstance(hand_reference_transforms_world, dict):
        raise CameraGeometryError("hand reference transforms must be an object")
    if set(hand_reference_transforms_world) != {
        "palm",
        "grip_point",
        "finger_roots",
    }:
        raise CameraGeometryError(
            "hand reference transforms require palm, grip_point, and finger_roots"
        )
    finger_roots = hand_reference_transforms_world["finger_roots"]
    if not isinstance(finger_roots, (list, tuple)) or len(finger_roots) != 2:
        raise CameraGeometryError(
            "hand reference transforms require exactly two finger roots"
        )

    palm = validated_rigid_transform(
        hand_reference_transforms_world["palm"],
        name="palm world transform",
    )
    grip_point = validated_rigid_transform(
        hand_reference_transforms_world["grip_point"],
        name="grip-point world transform",
    )
    fingers = [
        validated_rigid_transform(
            transform,
            name=f"finger-root-{index + 1} world transform",
        )
        for index, transform in enumerate(finger_roots)
    ]

    def distance_to(transform: np.ndarray) -> float:
        distance = float(np.linalg.norm(target_world - transform[:3, 3]))
        if not math.isfinite(distance) or distance < 0.0:
            raise CameraGeometryError("hand reference distance is invalid")
        return distance

    finger_distances = [distance_to(transform) for transform in fingers]
    return {
        "target_point_camera_xyz_m": effective_point.astype(float).tolist(),
        # This remains internal to the simulator response construction.  The
        # public contract exposes the frame-bound camera point and distances,
        # not an extra unreviewed world-space motion target.
        "target_point_world_xyz_m": target_world.astype(float).tolist(),
        "target_to_palm_m": distance_to(palm),
        "target_to_grip_point_m": distance_to(grip_point),
        "target_to_finger_roots_m": min(finger_distances),
        "target_to_finger_roots_individual_m": finger_distances,
    }


def project_world_to_pixel(
    frame: RgbdFrame, xyz_world: Any
) -> tuple[float, float, float]:
    world_to_camera = np.linalg.inv(frame.camera_to_world)
    point_camera = transform_point(world_to_camera, xyz_world)
    profile = frame.correction_profile
    if profile is not None and profile.enabled:
        point_camera = transform_point(
            np.linalg.inv(profile.raw_to_corrected_camera),
            point_camera,
        )
    u, v = pixel_from_camera_point(frame.intrinsics, point_camera)
    return u, v, -float(point_camera[2])


def _neighbor_world_point(
    frame: RgbdFrame, u: int, v: int, du: int, dv: int
) -> np.ndarray | None:
    uu = u + du
    vv = v + dv
    if (
        uu < 0
        or vv < 0
        or uu >= frame.intrinsics.width
        or vv >= frame.intrinsics.height
    ):
        return None
    depth = frame.depth_m[vv, uu]
    if not np.isfinite(depth) or depth <= 0:
        return None
    point_camera = camera_point_from_pixel(
        frame.intrinsics,
        u=uu,
        v=vv,
        depth_m=float(depth),
    )
    point_camera = frame.correction_profile.apply_camera_point(point_camera)
    return transform_point(frame.camera_to_world, point_camera)


def _surface_normal(
    frame: RgbdFrame, *, u: int, v: int, xyz_world: np.ndarray
) -> np.ndarray | None:
    left = _neighbor_world_point(frame, u, v, -1, 0)
    right = _neighbor_world_point(frame, u, v, 1, 0)
    up = _neighbor_world_point(frame, u, v, 0, -1)
    down = _neighbor_world_point(frame, u, v, 0, 1)
    if left is None or right is None or up is None or down is None:
        return None
    dx = right - left
    dy = down - up
    normal = np.cross(dx, dy)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    camera_origin = frame.camera_to_world[:3, 3]
    view_dir = camera_origin - xyz_world
    if float(np.dot(normal, view_dir)) < 0:
        normal = -normal
    return normal


def backproject_pixel_to_world(
    frame: RgbdFrame,
    *,
    u: Any,
    v: Any,
    depth_window_px: int = 7,
    output_frame: str = "world",
) -> dict[str, Any]:
    """Back-project a cached RGB-D pixel to world coordinates."""
    if output_frame != "world":
        raise CameraGeometryError("only output_frame='world' is currently supported")
    uu = _to_int_pixel("u", u)
    vv = _to_int_pixel("v", v)
    if int(depth_window_px) < 1:
        raise CameraGeometryError("depth_window_px must be >= 1")

    stats = _cluster_depth(frame, u=uu, v=vv, depth_window_px=int(depth_window_px))
    point_camera = camera_point_from_pixel(
        frame.intrinsics,
        u=uu,
        v=vv,
        depth_m=stats["depth_m"],
    )
    point_camera = frame.correction_profile.apply_camera_point(point_camera)
    xyz_world = transform_point(frame.camera_to_world, point_camera)
    reproj_u, reproj_v, reproj_depth = project_world_to_pixel(frame, xyz_world)
    reproj_error_px = math.hypot(reproj_u - float(uu), reproj_v - float(vv))
    if reproj_error_px > 1.0 + 1e-6:
        raise CameraGeometryError(
            f"projection round-trip error {reproj_error_px:.6f}px exceeds 1px"
        )
    normal = _surface_normal(frame, u=uu, v=vv, xyz_world=xyz_world)
    confidence = max(
        0.0,
        min(
            1.0,
            stats["cluster_ratio"]
            * (1.0 - min(1.0, stats["mad_m"] / max(stats["depth_m"], 1e-6) / 0.04)),
        ),
    )
    return {
        "xyz": xyz_world.tolist(),
        "surface_normal": normal.tolist() if normal is not None else None,
        "depth": {
            "median_m": stats["depth_m"],
            "mad_m": stats["mad_m"],
            "valid_ratio": stats["valid_ratio"],
            "cluster_ratio": stats["cluster_ratio"],
            "sample_count": stats["sample_count"],
            "valid_count": stats["valid_count"],
            "cluster_count": stats["cluster_count"],
        },
        "confidence": confidence,
        "reprojection_error_px": float(reproj_error_px),
        "reprojected_depth_m": float(reproj_depth),
    }


class FrameCache:
    """Small latest-frame cache that rejects stale frame IDs."""

    def __init__(
        self,
        *,
        max_frames_per_camera: int = 8,
        ttl_s: float = 30.0,
        correction_profiles: dict[str, CameraCorrectionProfile] | None = None,
    ) -> None:
        self.max_frames_per_camera = int(max_frames_per_camera)
        self.ttl_s = float(ttl_s)
        self._correction_profiles = {
            camera: CameraCorrectionProfile.identity(camera)
            for camera in ("head", "left_wrist", "right_wrist")
        }
        for camera, profile in (correction_profiles or {}).items():
            self.set_correction_profile(camera, profile)
        self._frames: dict[str, list[RgbdFrame]] = {}
        self._latest_capture_group_id: str | None = None

    def set_correction_profile(
        self,
        camera: str,
        profile: CameraCorrectionProfile,
    ) -> None:
        cam = canonical_camera(camera)
        if profile.camera != cam:
            raise CameraGeometryError(
                "correction profile camera does not match cache camera"
            )
        self._correction_profiles[cam] = profile

    def clear(self) -> None:
        """Invalidate every frame after an explicit simulator state restore."""

        self._frames.clear()
        self._latest_capture_group_id = None

    def add(
        self,
        *,
        camera: str,
        rgb: Any,
        depth_m: Any,
        intrinsics: CameraIntrinsics,
        camera_to_world: Any,
        step_index: int,
        timestamp_s: float | None = None,
        frame_id: str | None = None,
        capture_group_id: str | None = None,
        capture_metadata: dict[str, Any] | None = None,
    ) -> RgbdFrame:
        cam = canonical_camera(camera)
        frame = RgbdFrame(
            camera=cam,
            frame_id=frame_id or f"{cam}:{int(step_index)}:{uuid.uuid4().hex[:8]}",
            rgb=_as_array(rgb),
            depth_m=_normalize_depth(depth_m),
            intrinsics=intrinsics,
            camera_to_world=_homogeneous(camera_to_world),
            step_index=int(step_index),
            capture_group_id=capture_group_id,
            capture_metadata=dict(capture_metadata or {}),
            correction_profile=self._correction_profiles[cam],
            timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
        )
        bucket = self._frames.setdefault(cam, [])
        bucket.append(frame)
        del bucket[: -self.max_frames_per_camera]
        return frame

    def add_capture_group(
        self,
        *,
        frames: dict[str, dict[str, Any]],
        step_index: int,
        capture_metadata: dict[str, Any] | None = None,
        timestamp_s: float | None = None,
        capture_group_id: str | None = None,
    ) -> dict[str, RgbdFrame]:
        """Atomically add one same-sim-step capture for all planner cameras."""
        expected = {"head", "left_wrist", "right_wrist"}
        canonical: dict[str, dict[str, Any]] = {}
        for camera, values in frames.items():
            cam = canonical_camera(camera)
            if cam in canonical:
                raise CameraGeometryError(f"duplicate camera in capture group: {cam}")
            canonical[cam] = dict(values)
        if set(canonical) != expected:
            missing = sorted(expected - set(canonical))
            extra = sorted(set(canonical) - expected)
            raise CameraGeometryError(
                f"capture group must contain exactly head/left_wrist/right_wrist; "
                f"missing={missing}, extra={extra}"
            )

        captured_at = time.monotonic() if timestamp_s is None else float(timestamp_s)
        group_id = capture_group_id or (
            f"capture:{int(step_index)}:{uuid.uuid4().hex[:10]}"
        )
        metadata = deepcopy(dict(capture_metadata or {}))
        pending: dict[str, RgbdFrame] = {}
        # Construct every frame before mutating any bucket. A malformed camera
        # therefore leaves the previous three-camera group entirely intact.
        for cam in ("head", "left_wrist", "right_wrist"):
            values = canonical[cam]
            pending[cam] = RgbdFrame(
                camera=cam,
                frame_id=f"{cam}:{int(step_index)}:{group_id.rsplit(':', 1)[-1]}",
                rgb=_as_array(values["rgb"]),
                depth_m=_normalize_depth(values["depth_m"]),
                intrinsics=values["intrinsics"],
                camera_to_world=_homogeneous(values["camera_to_world"]),
                step_index=int(step_index),
                capture_group_id=group_id,
                capture_metadata=metadata,
                correction_profile=self._correction_profiles[cam],
                timestamp_s=captured_at,
            )
        for cam, frame in pending.items():
            bucket = self._frames.setdefault(cam, [])
            bucket.append(frame)
            del bucket[: -self.max_frames_per_camera]
        self._latest_capture_group_id = group_id
        return pending

    def latest(self, camera: str) -> RgbdFrame:
        cam = canonical_camera(camera)
        bucket = self._frames.get(cam) or []
        if not bucket:
            raise CameraGeometryError(f"no cached RGB-D frame for camera {cam}")
        return bucket[-1]

    def get_current(
        self, camera: str, frame_id: str, *, now_s: float | None = None
    ) -> RgbdFrame:
        cam = canonical_camera(camera)
        latest = self.latest(cam)
        if latest.frame_id != frame_id:
            raise CameraGeometryError(
                "stale frame_id; call observe() again before pixel_to_world"
            )
        if (
            latest.capture_group_id is not None
            and self._latest_capture_group_id != latest.capture_group_id
        ):
            raise CameraGeometryError(
                "stale capture group; call observe() again before pixel_to_world"
            )
        now = time.monotonic() if now_s is None else float(now_s)
        if now - latest.timestamp_s > self.ttl_s:
            raise CameraGeometryError(
                "frame_id expired; call observe() again before pixel_to_world"
            )
        return latest

    def observe_payload(self, camera: str) -> dict[str, Any]:
        frame = self.latest(camera)
        now = time.monotonic()
        age_s = now - frame.timestamp_s
        if age_s > self.ttl_s:
            raise CameraGeometryError(
                "latest RGB-D capture expired; a new simulator observation is required"
            )
        if (
            frame.capture_group_id is not None
            and self._latest_capture_group_id != frame.capture_group_id
        ):
            raise CameraGeometryError(
                "latest camera frame is not in the current capture group"
            )
        depth_image, depth_metadata = _depth_png_bytes(frame.depth_m)
        payload = {
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "capture_group": {
                "id": frame.capture_group_id,
                "sim_step": int(frame.step_index),
                "cameras": (
                    ["head", "left_wrist", "right_wrist"]
                    if frame.capture_group_id is not None
                    else [frame.camera]
                ),
                "age_s": float(max(0.0, age_s)),
            },
            "width": int(frame.intrinsics.width),
            "height": int(frame.intrinsics.height),
            "_image_bytes": _png_bytes(frame.rgb),
            "_depth_image_bytes": depth_image,
            "image_format": "png",
            "image_blocks": ["rgb", "depth_visualization"],
            "depth_visualization": depth_metadata,
            "correction": {
                "enabled": bool(frame.correction_profile.enabled),
                "metrics": dict(frame.correction_profile.metrics),
            },
        }
        # Keep the public observation boundary explicit. New simulator-side
        # metadata cannot become VLM-visible merely by being added to a frame.
        if "proprio" in frame.capture_metadata:
            payload["proprio"] = deepcopy(frame.capture_metadata["proprio"])
        return payload


__all__ = [
    "CameraGeometryError",
    "CameraCorrectionProfile",
    "CameraIntrinsics",
    "FrameCache",
    "RgbdFrame",
    "backproject_pixel_to_world",
    "camera_point_from_pixel",
    "canonical_camera",
    "correction_profile_to_json",
    "evaluate_camera_correction_profile",
    "fit_camera_correction_profile",
    "frame_bound_hand_distance_report",
    "load_camera_correction_profiles",
    "pixel_from_camera_point",
    "project_world_to_pixel",
    "r1pro_wrist_camera_reference_transforms",
    "rigid_transform_residual",
    "robust_depth_sample",
    "transform_point",
    "validated_rigid_transform",
]
