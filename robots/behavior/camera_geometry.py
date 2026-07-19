"""RGB-D frame cache and camera geometry for BEHAVIOR planner tools."""
from __future__ import annotations

import math
import time
import uuid
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
    if before_median <= 0 or not np.isfinite([before_median, before_p95, after_median, after_p95]).all():
        return False
    return (
        after_median <= before_median * (1.0 - float(min_median_improvement))
        and after_p95 <= before_p95
        and after_median <= float(max_final_median_error_m)
    )


def _sample_points(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    source_keys = ("raw_camera_xyz", "observed_camera_xyz", "measured_camera_xyz", "predicted_camera_xyz")
    target_keys = ("true_camera_xyz", "marker_camera_xyz", "target_camera_xyz", "ground_truth_camera_xyz")
    sources = []
    targets = []
    for sample in samples:
        source = next((sample[key] for key in source_keys if key in sample), None)
        target = next((sample[key] for key in target_keys if key in sample), None)
        if source is None or target is None:
            raise CameraGeometryError("calibration sample missing camera-frame source/target xyz")
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
    raw_to_corrected_camera: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    enabled: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", canonical_camera(self.camera))
        matrix = _homogeneous(self.raw_to_corrected_camera)
        if not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-3):
            raise CameraGeometryError("camera correction rotation must be a proper SE(3) rotation")
        object.__setattr__(self, "raw_to_corrected_camera", matrix)

    @classmethod
    def identity(cls, camera: str, *, reason: str = "identity_default") -> "CameraCorrectionProfile":
        return cls(
            camera=canonical_camera(camera),
            raw_to_corrected_camera=np.eye(4, dtype=np.float64),
            enabled=False,
            metrics={"enabled": False, "reason": reason},
        )

    @classmethod
    def from_mapping(cls, camera: str, value: dict[str, Any]) -> "CameraCorrectionProfile":
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
            return cls.identity(camera, reason="heldout_gate_failed_on_load").with_metrics(metrics)
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
    corrected = np.stack([profile.apply_camera_point(point) for point in source], axis=0)
    after = np.linalg.norm(corrected - target, axis=1)
    return {
        "samples": int(source.shape[0]),
        "before_median_m": float(np.median(before)),
        "before_p95_m": float(np.percentile(before, 95)),
        "after_median_m": float(np.median(after)),
        "after_p95_m": float(np.percentile(after, 95)),
        "translation_norm_m": float(np.linalg.norm(profile.raw_to_corrected_camera[:3, 3])),
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
    return CameraCorrectionProfile.identity(camera, reason="heldout_gate_failed").with_metrics(metrics)


def load_camera_correction_profiles(path: str | Path | None) -> dict[str, CameraCorrectionProfile]:
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
        raise CameraGeometryError("camera correction profile file must contain a mapping")
    for camera, value in raw_profiles.items():
        cam = canonical_camera(camera)
        if not isinstance(value, dict):
            raise CameraGeometryError(f"camera correction profile for {cam} must be an object")
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
    def from_image_shape(cls, shape: tuple[int, ...], *, fov_deg: float = 90.0) -> "CameraIntrinsics":
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
    correction_profile: CameraCorrectionProfile | None = None
    timestamp_s: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.camera = canonical_camera(self.camera)
        self.rgb = _as_array(self.rgb)
        self.depth_m = _normalize_depth(self.depth_m)
        if self.rgb.ndim != 3 or self.rgb.shape[2] not in (3, 4):
            raise CameraGeometryError(f"rgb image must be HxWx3/4, got {self.rgb.shape}")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise CameraGeometryError(
                f"depth shape {self.depth_m.shape} does not match rgb {self.rgb.shape[:2]}"
            )
        if self.intrinsics.width != self.rgb.shape[1] or self.intrinsics.height != self.rgb.shape[0]:
            raise CameraGeometryError("intrinsics dimensions do not match rgb image")
        self.intrinsics.validate()
        self.camera_to_world = _homogeneous(self.camera_to_world)
        if self.correction_profile is None:
            self.correction_profile = CameraCorrectionProfile.identity(self.camera)
        elif self.correction_profile.camera != self.camera:
            raise CameraGeometryError("correction profile camera does not match RGB-D frame camera")


def _normalize_depth(depth: Any) -> np.ndarray:
    array = _as_array(depth, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise CameraGeometryError(f"depth must be HxW, got {array.shape}")
    finite = array[np.isfinite(array) & (array > 0)]
    if finite.size and np.nanmedian(finite) > 20.0:
        array = array / 1000.0
    return array.astype(np.float64, copy=False)


def _png_bytes(rgb: Any) -> bytes:
    array = _as_array(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CameraGeometryError(f"rgb image must be HxWx3/4, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        finite = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and finite.max() <= 1.0:
            finite = finite * 255.0
        array = np.clip(finite, 0.0, 255.0).astype(np.uint8)
    buffer = BytesIO()
    try:
        import imageio.v2 as imageio

        imageio.imwrite(buffer, array, format="png")
    except Exception as exc:
        raise CameraGeometryError(f"failed to encode RGB observation as PNG: {exc}") from exc
    return buffer.getvalue()


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
        raise CameraGeometryError("pixel window touches image border; refusing edge projection")
    crop = frame.depth_m[v - radius : v + radius + 1, u - radius : u + radius + 1]
    valid = _valid_depth_values(crop)
    valid_ratio = float(valid.size / crop.size) if crop.size else 0.0
    if valid.size < max(3, crop.size // 5):
        raise CameraGeometryError("not enough valid depth samples around pixel")

    median = float(np.median(valid))
    p10, p90 = np.percentile(valid, [10, 90])
    if float(p90 - p10) > max(0.08, 0.12 * max(median, 1e-6)):
        raise CameraGeometryError("depth window has no stable foreground cluster")
    abs_dev = np.abs(valid - median)
    mad = float(np.median(abs_dev))
    scale = max(0.002, 2.5 * 1.4826 * mad)
    cluster = valid[abs_dev <= scale]
    if cluster.size < max(3, int(valid.size * 0.35)):
        raise CameraGeometryError("depth window has no stable foreground cluster")
    cluster_median = float(np.median(cluster))
    cluster_mad = float(np.median(np.abs(cluster - cluster_median)))
    if cluster_median <= 0 or not math.isfinite(cluster_median):
        raise CameraGeometryError("depth cluster produced invalid depth")
    if cluster_mad / max(cluster_median, 1e-6) > 0.04:
        raise CameraGeometryError("depth cluster is too dispersed for reliable projection")

    return {
        "depth_m": cluster_median,
        "mad_m": cluster_mad,
        "valid_ratio": valid_ratio,
        "cluster_ratio": float(cluster.size / crop.size),
        "sample_count": int(crop.size),
        "valid_count": int(valid.size),
        "cluster_count": int(cluster.size),
    }


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


def pixel_from_camera_point(intrinsics: CameraIntrinsics, point_camera: Any) -> tuple[float, float]:
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


def project_world_to_pixel(frame: RgbdFrame, xyz_world: Any) -> tuple[float, float, float]:
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


def _neighbor_world_point(frame: RgbdFrame, u: int, v: int, du: int, dv: int) -> np.ndarray | None:
    uu = u + du
    vv = v + dv
    if uu < 0 or vv < 0 or uu >= frame.intrinsics.width or vv >= frame.intrinsics.height:
        return None
    depth = frame.depth_m[vv, uu]
    if not np.isfinite(depth) or depth <= 0:
        return None
    return transform_point(
        frame.camera_to_world,
        camera_point_from_pixel(frame.intrinsics, u=uu, v=vv, depth_m=float(depth)),
    )


def _surface_normal(frame: RgbdFrame, *, u: int, v: int, xyz_world: np.ndarray) -> np.ndarray | None:
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

    def set_correction_profile(
        self,
        camera: str,
        profile: CameraCorrectionProfile,
    ) -> None:
        cam = canonical_camera(camera)
        if profile.camera != cam:
            raise CameraGeometryError("correction profile camera does not match cache camera")
        self._correction_profiles[cam] = profile

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
            correction_profile=self._correction_profiles[cam],
            timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
        )
        bucket = self._frames.setdefault(cam, [])
        bucket.append(frame)
        del bucket[:-self.max_frames_per_camera]
        return frame

    def latest(self, camera: str) -> RgbdFrame:
        cam = canonical_camera(camera)
        bucket = self._frames.get(cam) or []
        if not bucket:
            raise CameraGeometryError(f"no cached RGB-D frame for camera {cam}")
        return bucket[-1]

    def get_current(self, camera: str, frame_id: str, *, now_s: float | None = None) -> RgbdFrame:
        cam = canonical_camera(camera)
        latest = self.latest(cam)
        if latest.frame_id != frame_id:
            raise CameraGeometryError("stale frame_id; call observe() again before pixel_to_world")
        now = time.monotonic() if now_s is None else float(now_s)
        if now - latest.timestamp_s > self.ttl_s:
            raise CameraGeometryError("frame_id expired; call observe() again before pixel_to_world")
        return latest

    def observe_payload(self, camera: str) -> dict[str, Any]:
        frame = self.latest(camera)
        # Simulator state only changes through facade calls that add a new frame.
        # Reissue the unchanged latest capture so a just-returned frame_id can
        # always be consumed, while any ID returned by an earlier observe call
        # remains stale.
        frame.frame_id = (
            f"{frame.camera}:{int(frame.step_index)}:{uuid.uuid4().hex[:8]}"
        )
        frame.timestamp_s = time.monotonic()
        return {
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "width": int(frame.intrinsics.width),
            "height": int(frame.intrinsics.height),
            "_image_bytes": _png_bytes(frame.rgb),
            "image_format": "png",
            "correction": {
                "enabled": bool(frame.correction_profile.enabled),
                "metrics": dict(frame.correction_profile.metrics),
            },
        }


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
    "load_camera_correction_profiles",
    "pixel_from_camera_point",
    "project_world_to_pixel",
    "transform_point",
]
