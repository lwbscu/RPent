#!/usr/bin/env python3
"""Deterministic real-simulator acceptance for the BEHAVIOR planner surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np

from robots.behavior.camera_geometry import (
    backproject_pixel_to_world,
    camera_point_from_pixel,
    correction_profile_to_json,
    fit_camera_correction_profile,
    robust_depth_sample,
    transform_point,
)
from robots.behavior.env_server import (
    BehaviorEnvFacade,
    _load_env_config,
    _sensor_camera_to_world,
)
from robots.behavior.planner_executor import _artifact_jsonable

CAMERAS = ("head", "left_wrist", "right_wrist")
HANDS = ("left", "right")
POSITION_TOLERANCE_M = 0.02
ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
HELD_STEPS_REQUIRED = 10
MIN_NORMALIZED_JOINT_MARGIN = 0.03
PRIVATE_MARKER_DEPTH_CONSISTENCY_M = 0.005

_PRIVATE_MARKER_NAME = "rpent_private_camera_marker"
_PRIVATE_MARKER_RADIUS_M = 0.04
_PRIVATE_MARKER_CANDIDATES = (
    (0.00, 0.00, 0.45),
    (-0.06, -0.04, 0.50),
    (0.06, -0.04, 0.55),
    (-0.04, 0.05, 0.60),
    (0.04, 0.05, 0.65),
    (0.00, -0.06, 0.70),
    (-0.10, 0.00, 0.52),
    (0.10, 0.00, 0.58),
    (-0.07, 0.07, 0.64),
    (0.07, 0.07, 0.68),
    (-0.08, -0.07, 0.56),
    (0.08, -0.07, 0.62),
)
_PRIVATE_WRIST_MARKER_CANDIDATES = (
    (0.00, 0.00, 0.18),
    (-0.03, -0.02, 0.21),
    (0.03, -0.02, 0.24),
    (-0.02, 0.03, 0.27),
    (0.02, 0.03, 0.30),
    (0.00, -0.03, 0.33),
    (-0.05, 0.00, 0.20),
    (0.05, 0.00, 0.23),
    (-0.04, 0.04, 0.26),
    (0.04, 0.04, 0.29),
    (-0.04, -0.04, 0.22),
    (0.04, -0.04, 0.28),
)
_PRIVATE_PICK_FIXTURES = {
    "left": ("rpent_private_pick_left", [5.487, 5.586, 0.332]),
    "right": ("rpent_private_pick_right", [5.165, 4.949, 0.332]),
}
_PRIVATE_PRESS_FIXTURES = {
    "left": ("rpent_private_press_left", [5.487, 5.586, 0.33]),
    "right": ("rpent_private_press_right", [5.165, 4.949, 0.33]),
}
_PRIVATE_MANIPULATION_PRELOAD_POSITIONS = {
    "rpent_private_pick_left": [3.34, 5.20, 0.56],
    "rpent_private_pick_right": [3.70, 5.20, 0.56],
    "rpent_private_press_left": [3.43, 5.36, 0.54],
    "rpent_private_press_right": [3.61, 5.36, 0.54],
}
_PRIVATE_COLLISION_FIXTURE = "rpent_private_collision_obstacle"


def _install_private_fixture_config_passthrough() -> None:
    """Make acceptance-only ``omni_config.objects`` reach OmniGibson.

    RLinf intentionally merges only its standard env/robot/task/scene sections
    into OmniGibson's example config.  The deterministic acceptance fixtures
    are test-only objects, so this process-local wrapper adds exactly that one
    section before ``BehaviorEnv`` imports its copy of ``setup_omni_cfg``.  No
    installed RLinf source or production RPent environment path is modified.
    """

    from omegaconf import OmegaConf
    from rlinf.envs.behavior import utils as behavior_utils

    original = behavior_utils.setup_omni_cfg
    if getattr(original, "_rpent_private_fixture_passthrough", False):
        return

    def setup_with_private_fixtures(cfg: Any) -> Any:
        omni_cfg = original(cfg)
        objects = OmegaConf.select(cfg, "omni_config.objects", default=None)
        if objects:
            OmegaConf.update(omni_cfg, "objects", objects, merge=False)
        return omni_cfg

    setup_with_private_fixtures._rpent_private_fixture_passthrough = True
    behavior_utils.setup_omni_cfg = setup_with_private_fixtures


def _inject_private_marker_fixture(cfg: Any) -> None:
    """Preload one visual-only marker without changing public observations."""
    marker_cfg = {
        "type": "PrimitiveObject",
        "name": _PRIVATE_MARKER_NAME,
        "primitive_type": "Sphere",
        "radius": _PRIVATE_MARKER_RADIUS_M,
        "visual_only": True,
        "fixed_base": True,
        "include_default_states": False,
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "position": [100.0, 100.0, 100.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }
    objects = list(cfg.omni_config.get("objects") or [])
    objects.append(marker_cfg)
    cfg.omni_config.objects = objects


def _inject_private_manipulation_fixtures(cfg: Any) -> None:
    """Preload physical pick and guarded-contact fixtures for both hands."""
    objects = list(cfg.omni_config.get("objects") or [])
    for name, _position in _PRIVATE_PICK_FIXTURES.values():
        objects.append(
            {
                "type": "PrimitiveObject",
                "name": name,
                "primitive_type": "Cube",
                "scale": [0.03, 0.03, 0.03],
                "visual_only": False,
                "fixed_base": False,
                "rgba": [0.1, 0.7, 0.2, 1.0],
                "position": _PRIVATE_MANIPULATION_PRELOAD_POSITIONS[name],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
        )
    for name, _position in _PRIVATE_PRESS_FIXTURES.values():
        objects.append(
            {
                "type": "PrimitiveObject",
                "name": name,
                "primitive_type": "Cube",
                "scale": [0.04, 0.04, 0.02],
                "visual_only": False,
                "fixed_base": True,
                "rgba": [0.2, 0.2, 0.9, 1.0],
                "position": _PRIVATE_MANIPULATION_PRELOAD_POSITIONS[name],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
        )
    cfg.omni_config.objects = objects


def _inject_private_adversarial_fixture(cfg: Any) -> None:
    objects = list(cfg.omni_config.get("objects") or [])
    objects.append(
        {
            "type": "PrimitiveObject",
            "name": _PRIVATE_COLLISION_FIXTURE,
            "primitive_type": "Cube",
            "scale": [0.12, 0.12, 0.12],
            "visual_only": False,
            "fixed_base": True,
            "rgba": [0.9, 0.1, 0.1, 1.0],
            "position": [100.0, 100.0, 100.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
    )
    cfg.omni_config.objects = objects


def _ensure_private_marker(env: BehaviorEnvFacade) -> Any:
    scene = env._env.omnigibson_env.scene
    marker = scene.object_registry("name", _PRIVATE_MARKER_NAME)
    if marker is None:
        raise RuntimeError(
            "private marker was not loaded before reset; acceptance fixture "
            "config passthrough is unavailable"
        )
    return marker


def _ensure_private_manipulation_fixtures(env: BehaviorEnvFacade) -> None:
    scene = env._env.omnigibson_env.scene
    for name, _position in _PRIVATE_PICK_FIXTURES.values():
        obj = scene.object_registry("name", name)
        if obj is None:
            raise RuntimeError(
                f"private pick fixture was not loaded before reset: {name}"
            )
    for name, _position in _PRIVATE_PRESS_FIXTURES.values():
        obj = scene.object_registry("name", name)
        if obj is None:
            raise RuntimeError(
                f"private press fixture was not loaded before reset: {name}"
            )


def _place_private_fixture(
    env: BehaviorEnvFacade,
    name: str,
    position: Iterable[float],
    *,
    settle_steps: int,
) -> np.ndarray:
    """Activate one simulator-side fixture without moving the robot."""

    import omnigibson as og

    obj = env._env.omnigibson_env.scene.object_registry("name", name)
    if obj is None:
        raise RuntimeError(f"private manipulation fixture is unavailable: {name}")
    obj.set_position_orientation(
        position=list(position),
        orientation=[0.0, 0.0, 0.0, 1.0],
    )
    keep_still = getattr(obj, "keep_still", None)
    if callable(keep_still):
        keep_still()
    for _ in range(max(0, int(settle_steps))):
        og.sim.step()
    return np.asarray(obj.get_position_orientation()[0], dtype=np.float64)


def _deactivate_private_fixture(env: BehaviorEnvFacade, name: str) -> None:
    _place_private_fixture(
        env,
        name,
        [100.0, 100.0, 100.0],
        settle_steps=0,
    )


def _ensure_private_collision_fixture(env: BehaviorEnvFacade) -> Any:
    scene = env._env.omnigibson_env.scene
    obstacle = scene.object_registry("name", _PRIVATE_COLLISION_FIXTURE)
    if obstacle is None:
        raise RuntimeError("private collision fixture was not loaded before reset")
    return obstacle


def _place_marker_from_camera(
    env: BehaviorEnvFacade,
    *,
    camera: str,
    camera_xyz: Iterable[float],
) -> np.ndarray:
    import omnigibson as og

    sensor = env._sensor_for_camera(camera)
    placement_camera_to_world = (
        _sensor_camera_to_world(sensor) if sensor is not None else None
    )
    if placement_camera_to_world is None:
        raise RuntimeError(
            f"render-synchronous Kit camera transform unavailable for {camera}"
        )
    center_world = transform_point(
        placement_camera_to_world,
        np.asarray(list(camera_xyz), dtype=np.float64),
    )
    marker = env._env.omnigibson_env.scene.object_registry("name", _PRIVATE_MARKER_NAME)
    if marker is None:
        raise RuntimeError("private navigation marker was not loaded")
    marker.set_position_orientation(
        position=center_world,
        orientation=[0.0, 0.0, 0.0, 1.0],
    )
    # Replicator transform and label updates are asynchronous. Poll a small,
    # bounded number of render groups without advancing robot physics; each
    # candidate uses the official three render updates.
    last_error: Exception | None = None
    for _attempt in range(1, 6):
        og.sim.pi.update_simulation(
            elapsedStep=0,
            currentTime=og.sim.current_time,
        )
        for _ in range(3):
            og.sim.render()
        env._refresh_observation_without_step()
        try:
            u, v = _private_current_marker_pixel(env, camera=camera)
            _sphere_surface_camera_point(
                frame=env._frame_cache.latest(camera),
                u=u,
                v=v,
                center_world=np.asarray(marker.get_position_orientation()[0]),
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"private navigation marker did not stabilize for {camera}: {last_error}"
        )
    return center_world


def _private_current_marker_pixel(
    env: BehaviorEnvFacade,
    *,
    camera: str,
) -> tuple[int, int]:
    sensor = env._sensor_for_camera(camera)
    if sensor is None:
        raise RuntimeError(f"camera sensor unavailable for marker pixel: {camera}")
    if "seg_instance_id" not in sensor.modalities:
        raise RuntimeError("RGBWrapper did not initialize private seg_instance_id")
    private_obs, private_info = sensor.get_obs()
    try:
        return _marker_interior_pixel(
            private_obs["seg_instance_id"], private_info["seg_instance_id"]
        )
    except RuntimeError:
        candidates = _red_marker_interior_pixels(env._frame_cache.latest(camera).rgb)
        if not candidates:
            raise RuntimeError(
                "private marker is absent from segmentation and RGB marker mask"
            )
        return candidates[0]


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _marker_interior_pixel(
    segmentation: Any,
    labels: dict[Any, Any],
) -> tuple[int, int]:
    def belongs_to_marker(value: Any) -> bool:
        # ``seg_instance_id`` identifies the rendered mesh prim, not the
        # object root.  PrimitiveObject therefore normally appears as
        # ``.../<object-name>/base_link/visuals`` in the label registry.
        parts = str(value).rstrip("/").split("/")
        return _PRIVATE_MARKER_NAME in parts

    marker_ids = {int(key) for key, value in labels.items() if belongs_to_marker(value)}
    if not marker_ids:
        raise RuntimeError("private marker is absent from instance-segmentation labels")
    image = _as_numpy(segmentation).squeeze()
    mask = np.isin(image, list(marker_ids))
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise RuntimeError("private marker is not visible in instance segmentation")
    centroid = np.array([float(np.mean(rows)), float(np.mean(cols))])
    order = np.argsort(
        (rows.astype(np.float64) - centroid[0]) ** 2
        + (cols.astype(np.float64) - centroid[1]) ** 2
    )
    radius = 3
    height, width = mask.shape
    for index in order:
        v, u = int(rows[index]), int(cols[index])
        if v < radius or u < radius or v + radius >= height or u + radius >= width:
            continue
        if bool(mask[v - radius : v + radius + 1, u - radius : u + radius + 1].all()):
            return u, v
    raise RuntimeError("private marker has no stable 7x7 interior depth window")


def _red_marker_interior_pixels(rgb: Any) -> list[tuple[int, int]]:
    """Return stable interior pixels for the acceptance-only red sphere."""

    image = _as_numpy(rgb).astype(np.float64)
    if image.ndim != 3 or image.shape[-1] < 3:
        return []
    image = image[..., :3]
    if float(np.nanmax(image, initial=0.0)) > 1.5:
        image /= 255.0
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    mask = (
        np.isfinite(image).all(axis=-1)
        & (red >= 0.45)
        & (red - green >= 0.2)
        & (red - blue >= 0.2)
    )
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []
    centroid = np.array([float(np.mean(rows)), float(np.mean(cols))])
    order = np.argsort(
        (rows.astype(np.float64) - centroid[0]) ** 2
        + (cols.astype(np.float64) - centroid[1]) ** 2
    )
    radius = 3
    height, width = mask.shape
    result: list[tuple[int, int]] = []
    for index in order:
        v, u = int(rows[index]), int(cols[index])
        if v < radius or u < radius or v + radius >= height or u + radius >= width:
            continue
        if bool(mask[v - radius : v + radius + 1, u - radius : u + radius + 1].all()):
            result.append((u, v))
            if len(result) >= 512:
                break
    return result


def _sphere_surface_camera_point(
    *,
    frame: Any,
    u: int,
    v: int,
    center_world: np.ndarray,
) -> np.ndarray:
    ray = camera_point_from_pixel(frame.intrinsics, u=u, v=v, depth_m=1.0)
    ray /= np.linalg.norm(ray)
    center_camera = transform_point(np.linalg.inv(frame.camera_to_world), center_world)
    along = float(np.dot(ray, center_camera))
    discriminant = along * along - (
        float(np.dot(center_camera, center_camera)) - _PRIVATE_MARKER_RADIUS_M**2
    )
    if along <= 0.0 or discriminant <= 0.0:
        raise RuntimeError(
            "selected marker pixel ray does not intersect marker sphere: "
            f"camera={frame.camera} frame_id={frame.frame_id} pixel_uv={[u, v]} "
            f"center_world={np.asarray(center_world).tolist()} "
            f"center_camera={center_camera.tolist()} ray={ray.tolist()} "
            f"along={along:.9f} discriminant={discriminant:.9f} "
            f"camera_to_world={frame.camera_to_world.tolist()}"
        )
    distance = along - math.sqrt(discriminant)
    if distance <= 0.0:
        raise RuntimeError("marker sphere intersection is behind the camera")
    return ray * distance


def _point_set_rank(points: Iterable[Iterable[float]]) -> int:
    """Return the affine rank of a private marker sample set."""

    array = np.asarray(list(points), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 3:
        return 0
    centered = array - np.mean(array, axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered, tol=1e-6))


def _capture_live_marker_fixture(env: BehaviorEnvFacade) -> dict[str, Any]:
    """Collect private train/heldout truth from the live simulator scene."""
    import omnigibson as og

    og_env = env._env.omnigibson_env
    marker = og_env.scene.object_registry("name", _PRIVATE_MARKER_NAME)
    if marker is None:
        raise RuntimeError("private calibration marker was not loaded into the scene")
    cameras: dict[str, Any] = {}
    for camera in CAMERAS:
        sensor = env._sensor_for_camera(camera)
        if sensor is None:
            raise RuntimeError(
                f"camera sensor unavailable for private marker: {camera}"
            )
        if "seg_instance_id" not in sensor.modalities:
            raise RuntimeError("RGBWrapper did not initialize private seg_instance_id")
        samples = {
            "train_samples": [],
            "heldout_samples": [],
            "rejected_candidates": [],
        }
        try:
            marker_candidates = (
                _PRIVATE_MARKER_CANDIDATES
                if camera == "head"
                else _PRIVATE_WRIST_MARKER_CANDIDATES
            )
            for candidate_index, (offset_x, offset_y, depth_m) in enumerate(
                marker_candidates
            ):
                if len(samples["heldout_samples"]) >= 3:
                    break
                split = "train" if len(samples["train_samples"]) < 3 else "heldout"
                env._refresh_observation_without_step()
                placement_camera_to_world = _sensor_camera_to_world(sensor)
                if placement_camera_to_world is None:
                    raise RuntimeError(
                        "render-synchronous Kit camera transform unavailable for "
                        f"{camera}"
                    )
                desired_camera = np.array(
                    [float(offset_x), float(offset_y), -float(depth_m)],
                    dtype=np.float64,
                )
                center_world = transform_point(
                    placement_camera_to_world,
                    desired_camera,
                )
                marker.set_position_orientation(
                    position=center_world,
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
                # Replicator may expose either the previous transform or a
                # temporarily incomplete label registry. Poll render-only
                # groups until the labelled pixel geometrically agrees with
                # the current simulator marker; fail closed after five.
                last_error: Exception | None = None
                for propagation_attempt in range(1, 6):
                    og.sim.pi.update_simulation(
                        elapsedStep=0,
                        currentTime=og.sim.current_time,
                    )
                    for _ in range(3):
                        og.sim.render()
                    env._refresh_observation_without_step()
                    frame = env._frame_cache.latest(camera)
                    private_obs, private_info = sensor.get_obs()
                    try:
                        try:
                            candidates = [
                                _marker_interior_pixel(
                                    private_obs["seg_instance_id"],
                                    private_info["seg_instance_id"],
                                )
                            ]
                            marker_pixel_source = "private_seg_instance_id"
                        except RuntimeError:
                            candidates = _red_marker_interior_pixels(frame.rgb)
                            marker_pixel_source = "private_rgb_material_mask"
                        if not candidates:
                            raise RuntimeError(
                                "marker absent from segmentation and RGB material mask"
                            )
                        candidate_errors: list[str] = []
                        for u, v in candidates:
                            try:
                                true_camera_xyz = _sphere_surface_camera_point(
                                    frame=frame,
                                    u=u,
                                    v=v,
                                    center_world=np.asarray(
                                        marker.get_position_orientation()[0]
                                    ),
                                )
                                break
                            except RuntimeError as exc:
                                candidate_errors.append(str(exc))
                        else:
                            raise RuntimeError(
                                "no labelled/material marker pixel intersects the "
                                f"simulator sphere; first_errors={candidate_errors[:3]}"
                            )
                        depth_stats = robust_depth_sample(
                            frame,
                            u=u,
                            v=v,
                            window_px=7,
                        )
                        raw_camera_xyz = camera_point_from_pixel(
                            frame.intrinsics,
                            u=u,
                            v=v,
                            depth_m=float(depth_stats["depth_m"]),
                        )
                        paired_error_m = float(
                            np.linalg.norm(raw_camera_xyz - true_camera_xyz)
                        )
                        if paired_error_m > PRIVATE_MARKER_DEPTH_CONSISTENCY_M:
                            raise RuntimeError(
                                "rendered marker depth is stale relative to the current "
                                f"simulator sphere: paired_error_m={paired_error_m:.9f}"
                            )
                        break
                    except (RuntimeError, ValueError) as exc:
                        last_error = exc
                else:
                    samples["rejected_candidates"].append(
                        {
                            "candidate_index": candidate_index,
                            "desired_camera_xyz": [
                                offset_x,
                                offset_y,
                                -depth_m,
                            ],
                            "reason": f"{type(last_error).__name__}: {last_error}",
                        }
                    )
                    continue
                if split == "train" and len(samples["train_samples"]) == 2:
                    candidate_rank = _point_set_rank(
                        [
                            *(
                                sample["raw_camera_xyz"]
                                for sample in samples["train_samples"]
                            ),
                            raw_camera_xyz,
                        ]
                    )
                    if candidate_rank < 2:
                        samples["rejected_candidates"].append(
                            {
                                "candidate_index": candidate_index,
                                "desired_camera_xyz": desired_camera.tolist(),
                                "reason": (
                                    "training marker geometry is affine-rank deficient: "
                                    f"rank={candidate_rank}"
                                ),
                            }
                        )
                        continue
                projection = env.pixel_to_world(
                    camera=camera,
                    frame_id=frame.frame_id,
                    u=u,
                    v=v,
                    depth_window_px=7,
                )
                if not _projection_succeeded(projection):
                    raise RuntimeError(
                        f"public marker projection failed for {camera}/{split}: {projection}"
                    )
                samples[f"{split}_samples"].append(
                    {
                        "raw_camera_xyz": raw_camera_xyz.tolist(),
                        "true_camera_xyz": true_camera_xyz.tolist(),
                        "candidate_index": candidate_index,
                        "desired_camera_xyz": desired_camera.tolist(),
                        "pixel_uv": [u, v],
                        "frame_id": frame.frame_id,
                        "capture_group": frame.capture_group_id,
                        "sim_step": frame.step_index,
                        "marker_pose_propagation_attempts": propagation_attempt,
                        "marker_pixel_source": marker_pixel_source,
                        "depth_statistics": depth_stats,
                        "public_reprojection_error_px": float(
                            projection["metrics"]["reprojection_error_px"]
                        ),
                    }
                )
            if len(samples["train_samples"]) < 3 or len(samples["heldout_samples"]) < 3:
                raise RuntimeError(
                    f"insufficient visible marker correspondences for {camera}: "
                    f"train={len(samples['train_samples'])} "
                    f"heldout={len(samples['heldout_samples'])} "
                    f"rejected={samples['rejected_candidates']}"
                )
        finally:
            marker.set_position_orientation(
                position=[100.0, 100.0, 100.0],
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
        cameras[camera] = samples
    return {
        "source": "live_simulator_marker",
        "marker": {
            "name": _PRIVATE_MARKER_NAME,
            "primitive": "Sphere",
            "radius_m": _PRIVATE_MARKER_RADIUS_M,
            "visual_only": True,
        },
        "cameras": cameras,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-instance-id", type=int, default=211)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--phase",
        choices=(
            "perception",
            "motion",
            "navigation",
            "manipulation",
            "adversarial",
            "calibration",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--move-targets-per-hand", type=int, default=20)
    parser.add_argument(
        "--marker-calibration-json",
        help=(
            "Private simulator-marker train/heldout correspondences. The JSON "
            "must identify source='simulator_marker' and contain a mapping per camera."
        ),
    )
    parser.add_argument(
        "--stall-target-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Private real-simulator fixture target known to produce stalled_tracking. "
            "Without it, the stall acceptance is an explicit mandatory failure."
        ),
    )
    return parser.parse_args()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def _valid_pixel(frame: Any, *, window: int = 7) -> tuple[int, int]:
    height, width = frame.depth_m.shape
    center_u, center_v = width // 2, height // 2
    candidates = [(center_u, center_v)]
    for radius in range(20, min(width, height) // 2, 20):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            candidates.append(
                (
                    int(round(center_u + radius * math.cos(angle))),
                    int(round(center_v + radius * math.sin(angle))),
                )
            )
    for u, v in candidates:
        try:
            backproject_pixel_to_world(
                frame,
                u=u,
                v=v,
                depth_window_px=window,
            )
            return u, v
        except Exception:
            continue
    raise RuntimeError(f"no stable projection pixel found for {frame.camera}")


def _copy_capture_frames(env: BehaviorEnvFacade) -> dict[str, dict[str, Any]]:
    frames = {}
    for camera in CAMERAS:
        frame = env._frame_cache.latest(camera)
        frames[camera] = {
            "rgb": frame.rgb.copy(),
            "depth_m": frame.depth_m.copy(),
            "intrinsics": frame.intrinsics,
            "camera_to_world": frame.camera_to_world.copy(),
        }
    return frames


def _inject_negative_depth(
    env: BehaviorEnvFacade,
    *,
    u: int,
    v: int,
    mode: str,
) -> str:
    frames = _copy_capture_frames(env)
    depth = frames["head"]["depth_m"]
    radius = 3
    crop = depth[v - radius : v + radius + 1, u - radius : u + radius + 1]
    if mode == "invalid":
        crop[:] = np.nan
    elif mode == "edge":
        valid = depth[np.isfinite(depth) & (depth > 0)]
        baseline = float(np.median(valid))
        crop[:, :4] = max(0.05, baseline * 0.5)
        crop[:, 4:] = baseline * 1.5
    else:
        raise ValueError(mode)
    added = env._frame_cache.add_capture_group(
        frames=frames,
        step_index=env._env_steps,
        capture_metadata={"negative_control": mode},
    )
    return added["head"].frame_id


def _motion_targets(origin: np.ndarray, count: int) -> list[np.ndarray]:
    targets = []
    for index in range(count):
        angle = 2.0 * math.pi * index / max(1, count)
        targets.append(
            origin
            + np.array(
                [
                    0.012 * math.cos(angle),
                    0.010 * math.sin(angle),
                    0.006 * math.sin(2.0 * angle),
                ]
            )
        )
    return targets


def _primitive_succeeded(value: Any) -> bool:
    return isinstance(value, dict) and value.get("primitive_success") is True


def _projection_succeeded(value: Any) -> bool:
    if not _primitive_succeeded(value):
        return False
    try:
        return float(value["metrics"]["reprojection_error_px"]) <= 1.0 + 1e-6
    except (KeyError, TypeError, ValueError):
        return False


def _projection_rejected_with(value: Any, tokens: Iterable[str]) -> bool:
    if not isinstance(value, dict) or value.get("primitive_success") is not False:
        return False
    if value.get("stop_reason") != "projection_failed":
        return False
    error = str((value.get("diagnostics") or {}).get("error", "")).lower()
    return any(str(token).lower() in error for token in tokens)


def _joint_margin_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    margin = (value.get("metrics") or {}).get("joint_margin")
    if not isinstance(margin, dict) or margin.get("available") is not True:
        return False
    if margin.get("ok") is not True:
        return False
    try:
        raw_margin = float(margin["min_raw_margin_joint_units"])
        range_fraction = float(margin["min_range_fraction"])
    except (KeyError, TypeError, ValueError):
        try:
            return float(margin["min_normalized_margin"]) >= MIN_NORMALIZED_JOINT_MARGIN
        except (KeyError, TypeError, ValueError):
            return False
    return raw_margin >= 0.05 or range_fraction >= MIN_NORMALIZED_JOINT_MARGIN


def _dynamics_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    metrics = value.get("metrics") or {}
    command = metrics.get("dynamics")
    if not isinstance(command, dict):
        return False
    try:
        command_ok = (
            float(command["max_velocity_command_delta"])
            <= float(command["velocity_limit"]) + 1e-6
            and float(command["max_acceleration_command_delta"])
            <= float(command["acceleration_limit"]) + 1e-6
        )
    except (KeyError, TypeError, ValueError):
        return False
    actual_peak = command.get("actual_peak")
    if isinstance(actual_peak, dict) and actual_peak.get("available") is True:
        return command_ok and actual_peak.get("ok") is True
    trace = (value.get("diagnostics") or {}).get("trace")
    if not isinstance(trace, list) or not trace:
        return False
    actual_samples = [
        step.get("actual_dynamics")
        for step in trace
        if isinstance(step, dict) and isinstance(step.get("actual_dynamics"), dict)
    ]
    available_samples = [
        sample for sample in actual_samples if sample.get("available") is True
    ]
    return (
        bool(available_samples)
        and command_ok
        and all(sample.get("ok") is True for sample in available_samples)
    )


def _collision_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    report = (value.get("metrics") or {}).get("collision_report")
    return (
        isinstance(report, dict)
        and report.get("available") is True
        and report.get("colliding") is False
    )


def _motion_quality_ok(value: Any) -> bool:
    if not _primitive_succeeded(value):
        return False
    metrics = value.get("metrics") or {}
    try:
        position_ok = float(value["position_error_m"]) <= POSITION_TOLERANCE_M
        orientation_ok = (
            float(value["orientation_error_rad"]) <= ORIENTATION_TOLERANCE_RAD
        )
        held_ok = int(metrics["held_steps"]) >= HELD_STEPS_REQUIRED
    except (KeyError, TypeError, ValueError):
        return False
    return (
        position_ok
        and orientation_ok
        and held_ok
        and _joint_margin_ok(value)
        and _collision_ok(value)
        and _dynamics_ok(value)
    )


def _bounded_failure(
    value: Any,
    *,
    stop_reasons: Iterable[str],
) -> bool:
    return (
        isinstance(value, dict)
        and value.get("primitive_success") is False
        and str(value.get("stop_reason")) in set(stop_reasons)
        and value.get("recoverable") in {True, False}
    )


def _explicit_failure(reason: str, *, detail: str) -> dict[str, Any]:
    """Represent a missing real fixture as a hard failure, never as a pass."""

    return {
        "primitive_success": False,
        "task_success": False,
        "stop_reason": reason,
        "recoverable": False,
        "diagnostics": {"error": detail},
    }


def main() -> None:
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_args = SimpleNamespace(
        suite="behavior_2025_challenge",
        task=0,
        task_name="turning_on_radio",
        activity_definition_id=0,
        activity_instance_id=args.activity_instance_id,
        activity_instance_dir=args.activity_instance_dir,
        scene_model="house_double_floor_lower",
        seed=args.seed,
        max_episode_steps=24756,
        output_dir=str(output_dir),
        config_path=None,
        control_mode="planner_tools",
    )
    report_path = output_dir / "planner_acceptance.json"
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "initializing",
        "phase": args.phase,
        "process": {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "ports": [],
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "commit": _git(repo, "rev-parse", "HEAD"),
        "worktree_dirty": bool(_git(repo, "status", "--porcelain")),
        "configuration": vars(env_args),
        "harness_configuration": {
            "phase": args.phase,
            "move_targets_per_hand": args.move_targets_per_hand,
            "marker_calibration_json": (
                str(Path(args.marker_calibration_json).expanduser().resolve())
                if args.marker_calibration_json
                else None
            ),
            "stall_target_configured": args.stall_target_xyz is not None,
            "private_marker_pose_propagation_max_render_updates": (
                15 * len(CAMERAS) * len(_PRIVATE_MARKER_CANDIDATES)
                if args.phase in {"calibration", "all"}
                else (15 * len(HANDS) if args.phase in {"navigation", "all"} else 0)
            ),
            "private_marker_zero_physics_updates_max": (
                5 * len(CAMERAS) * len(_PRIVATE_MARKER_CANDIDATES)
                if args.phase in {"calibration", "all"}
                else (5 * len(HANDS) if args.phase in {"navigation", "all"} else 0)
            ),
        },
        "acceptance_thresholds": {
            "projection_roundtrip_px": 1.0,
            "move_successes_per_hand": 19,
            "position_error_m": POSITION_TOLERANCE_M,
            "orientation_error_rad": ORIENTATION_TOLERANCE_RAD,
            "held_steps": HELD_STEPS_REQUIRED,
            "joint_margin_normalized": MIN_NORMALIZED_JOINT_MARGIN,
        },
        "results": {},
        "mandatory_failures": [],
    }

    def save() -> None:
        report_path.write_text(
            json.dumps(_artifact_jsonable(report), indent=2),
            encoding="utf-8",
        )

    def record(
        name: str,
        call: Callable[[], Any],
        *,
        predicate: Callable[[Any], bool] | None = None,
        predicate_description: str = "call completed",
        expected_stop_reasons: Iterable[str] | None = None,
        elapsed_bound_s: float | None = None,
        mandatory: bool = True,
    ) -> Any:
        started = time.monotonic()
        harness_error = None
        try:
            value = call()
        except Exception as exc:
            harness_error = f"{type(exc).__name__}: {exc}"
            value = {"harness_error": f"{type(exc).__name__}: {exc}"}
        elapsed_s = round(time.monotonic() - started, 3)
        expected_values = (
            (expected_stop_reasons,)
            if isinstance(expected_stop_reasons, str)
            else (expected_stop_reasons or [])
        )
        expected = sorted({str(reason) for reason in expected_values})
        checks: dict[str, bool] = {"call_completed": harness_error is None}
        if predicate is not None:
            try:
                checks["predicate"] = bool(predicate(value))
            except Exception as exc:
                checks["predicate"] = False
                harness_error = (
                    harness_error or f"predicate {type(exc).__name__}: {exc}"
                )
        if expected:
            checks["expected_stop_reason"] = (
                isinstance(value, dict) and str(value.get("stop_reason")) in expected
            )
        if elapsed_bound_s is not None:
            checks["elapsed_bound"] = elapsed_s <= float(elapsed_bound_s)
        passed = all(checks.values())
        report["results"][name] = {
            "mandatory": bool(mandatory),
            "passed": bool(passed),
            "predicate": predicate_description,
            "expected_stop_reasons": expected,
            "elapsed_bound_s": elapsed_bound_s,
            "elapsed_s": elapsed_s,
            "checks": checks,
            "harness_error": harness_error,
            "value": value,
        }
        if mandatory and not passed and name not in report["mandatory_failures"]:
            report["mandatory_failures"].append(name)
        save()
        return value

    save()
    env: BehaviorEnvFacade | None = None
    try:
        cfg = _load_env_config(env_args)
        if args.phase in {"navigation", "calibration", "all"}:
            _inject_private_marker_fixture(cfg)
        if args.phase in {"manipulation", "all"}:
            _inject_private_manipulation_fixtures(cfg)
        if args.phase in {"adversarial", "all"}:
            _inject_private_adversarial_fixture(cfg)
        _install_private_fixture_config_passthrough()
        env = BehaviorEnvFacade(
            cfg=cfg,
            meta=vars(env_args),
            output_dir=output_dir,
            control_mode="planner_tools",
        )
        reset_observation, reset_info = env.reset()
        report["status"] = "running"
        report["reset"] = {
            "state_dimension": int(np.asarray(reset_observation["states"]).size),
            "official_task_success": bool(
                isinstance(reset_info, dict)
                and isinstance(reset_info.get("done"), dict)
                and reset_info["done"].get("success", False)
            ),
        }
        if args.phase in {"navigation", "calibration", "all"}:
            _ensure_private_marker(env)
        if args.phase in {"manipulation", "all"}:
            _ensure_private_manipulation_fixtures(env)
            report["harness_configuration"]["fixture_activation"] = (
                "one target at a time after cuRobo warmup"
            )
        if args.phase in {"adversarial", "all"}:
            _ensure_private_collision_fixture(env)
        save()

        if args.phase in {"perception", "all"}:
            observations = {
                camera: record(
                    f"observe_{camera}",
                    lambda camera=camera: env.observe(camera),
                    predicate=_primitive_succeeded,
                    predicate_description="public observe succeeds",
                )
                for camera in CAMERAS
            }
            sync_result = {
                "capture_group_ids": {
                    camera: value.get("capture_group", {}).get("id")
                    for camera, value in observations.items()
                },
                "sim_steps": {
                    camera: value.get("capture_group", {}).get("sim_step")
                    for camera, value in observations.items()
                },
                "all_three_synchronized": len(
                    {
                        value.get("capture_group", {}).get("id")
                        for value in observations.values()
                    }
                )
                == 1,
            }
            sync_result["all_sim_steps_equal"] = (
                len(set(sync_result["sim_steps"].values())) == 1
            )
            sync_result["all_cameras_present"] = set(observations) == set(CAMERAS)
            sync_result["all_ids_present"] = all(
                value is not None for value in sync_result["capture_group_ids"].values()
            )
            sync_result["all_steps_present"] = all(
                value is not None for value in sync_result["sim_steps"].values()
            )
            record(
                "perception_three_camera_sync",
                lambda sync_result=sync_result: sync_result,
                predicate=lambda value: all(
                    bool(value[key])
                    for key in (
                        "all_three_synchronized",
                        "all_sim_steps_equal",
                        "all_cameras_present",
                        "all_ids_present",
                        "all_steps_present",
                    )
                ),
                predicate_description=(
                    "head/left_wrist/right_wrist share one capture group and simulator step"
                ),
            )
            for camera in CAMERAS:
                frame = env._frame_cache.latest(camera)
                u, v = _valid_pixel(frame)
                record(
                    f"projection_{camera}",
                    lambda camera=camera, frame=frame, u=u, v=v: env.pixel_to_world(
                        camera=camera,
                        frame_id=frame.frame_id,
                        u=u,
                        v=v,
                    ),
                    predicate=_projection_succeeded,
                    predicate_description="public backprojection round-trip is <=1px",
                )
            old_frame_id = env._frame_cache.latest("head").frame_id
            env._refresh_observation_without_step()
            record(
                "projection_stale_frame_rejection",
                lambda: env.pixel_to_world(
                    camera="head",
                    frame_id=old_frame_id,
                    u=320,
                    v=240,
                ),
                predicate=lambda value: _projection_rejected_with(
                    value, ("stale frame_id",)
                ),
                predicate_description="stale frame_id is rejected precisely",
                expected_stop_reasons=("projection_failed",),
            )
            current = env._frame_cache.latest("head")
            u, v = _valid_pixel(current)
            invalid_id = _inject_negative_depth(env, u=u, v=v, mode="invalid")
            record(
                "projection_invalid_depth_rejection",
                lambda: env.pixel_to_world(
                    camera="head", frame_id=invalid_id, u=u, v=v
                ),
                predicate=lambda value: _projection_rejected_with(
                    value,
                    ("not enough valid depth", "invalid depth", "no stable foreground"),
                ),
                predicate_description="invalid/non-finite depth is rejected precisely",
                expected_stop_reasons=("projection_failed",),
            )
            env._refresh_observation_without_step()
            current = env._frame_cache.latest("head")
            u, v = _valid_pixel(current)
            edge_id = _inject_negative_depth(env, u=u, v=v, mode="edge")
            record(
                "projection_depth_edge_rejection",
                lambda: env.pixel_to_world(camera="head", frame_id=edge_id, u=u, v=v),
                predicate=lambda value: _projection_rejected_with(
                    value,
                    ("stable foreground cluster", "too dispersed", "depth edge"),
                ),
                predicate_description="mixed depth edge is rejected precisely",
                expected_stop_reasons=("projection_failed",),
            )
            env._refresh_observation_without_step()
            save()

        if args.phase in {"motion", "all"}:
            for hand in HANDS:
                record(
                    f"release_{hand}_initial",
                    lambda hand=hand: env.release(
                        hand=hand,
                        opening=1.0,
                        retreat_m=0.0,
                    ),
                    predicate=_primitive_succeeded,
                    predicate_description="public release succeeds",
                )
                origin, target_quat = env._planner.backend.get_eef_pose(hand)
                results = []
                for index, target in enumerate(
                    _motion_targets(origin, args.move_targets_per_hand)
                ):
                    results.append(
                        record(
                            f"move_{hand}_{index:02d}",
                            lambda hand=hand, target=target: env.move_to(
                                hand=hand,
                                target_xyz=target.tolist(),
                                target_quat_xyzw=np.asarray(target_quat).tolist(),
                                timeout_s=45.0,
                            ),
                            predicate=_motion_quality_ok,
                            predicate_description=(
                                "move succeeds at <=2cm/5deg for 10 holds with safe joints/dynamics"
                            ),
                            mandatory=False,
                        )
                    )
                move_summary = {
                    "target_count": len(results),
                    "quality_success_count": sum(
                        _motion_quality_ok(result) for result in results
                    ),
                    "required_target_count": 20,
                    "required_quality_successes": 19,
                    "max_position_error_m": max(
                        (
                            float(result["position_error_m"])
                            for result in results
                            if result.get("position_error_m") is not None
                        ),
                        default=None,
                    ),
                    "max_orientation_error_rad": max(
                        (
                            float(result["orientation_error_rad"])
                            for result in results
                            if result.get("orientation_error_rad") is not None
                        ),
                        default=None,
                    ),
                }
                record(
                    f"move_{hand}_summary",
                    lambda move_summary=move_summary: move_summary,
                    predicate=lambda value: (
                        int(value["target_count"]) >= 20
                        and int(value["quality_success_count"]) >= 19
                    ),
                    predicate_description="at least 19 of 20 real targets satisfy every quality gate",
                )
                record(
                    f"rotate_{hand}",
                    lambda hand=hand: env.rotate_wrist(
                        hand=hand,
                        relative_axis_angle=[0.0, 0.0, 1.0, math.radians(3.0)],
                        frame="eef",
                    ),
                    predicate=_motion_quality_ok,
                    predicate_description=(
                        "rotate_wrist succeeds at <=2cm/5deg for 10 holds with safe joints/dynamics"
                    ),
                )
                record(
                    f"release_{hand}",
                    lambda hand=hand: env.release(
                        hand=hand,
                        opening=1.0,
                        retreat_m=0.0,
                    ),
                    predicate=_primitive_succeeded,
                    predicate_description="public release succeeds",
                )
                save()
            record(
                "move_unreachable_bounded",
                lambda: env.move_to(
                    hand="left",
                    target_xyz=[100.0, 100.0, 100.0],
                    timeout_s=2.0,
                ),
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=("navigation_required", "unreachable", "timeout"),
                ),
                predicate_description="unreachable move fails recoverably within a bound",
                expected_stop_reasons=("navigation_required", "unreachable", "timeout"),
                elapsed_bound_s=10.0,
            )
            record(
                "move_unreachable_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after unreachable move",
            )
            record(
                "move_timeout_bounded",
                lambda: env.move_to(
                    hand="right",
                    target_xyz=[6.0, 5.0, 1.0],
                    timeout_s=0.001,
                ),
                predicate=lambda value: _bounded_failure(
                    value, stop_reasons=("timeout",)
                ),
                predicate_description="move timeout fails recoverably within a bound",
                expected_stop_reasons=("timeout",),
                elapsed_bound_s=5.0,
            )
            record(
                "move_timeout_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after move timeout",
            )

        if args.phase in {"navigation", "all"}:
            surface_target = np.asarray([6.4258046597, 4.9026785517, 0.9806372991])
            record(
                "navigate_known_surface",
                lambda: env.navigate_to(
                    hand="left",
                    target_xyz=surface_target.tolist(),
                    standoff_m=0.85,
                    timeout_s=90.0,
                ),
                predicate=_primitive_succeeded,
                predicate_description="collision-checked cuRobo BASE navigation succeeds",
                elapsed_bound_s=95.0,
            )
            record(
                "navigate_unreachable_bounded",
                lambda: env.navigate_to(
                    hand="left",
                    target_xyz=[100.0, 100.0, 1.0],
                    timeout_s=5.0,
                ),
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=(
                        "navigation_unreachable",
                        "navigation_collision",
                        "base_plan_failed",
                        "timeout",
                    ),
                ),
                predicate_description="unreachable navigation fails recoverably within a bound",
                expected_stop_reasons=(
                    "navigation_unreachable",
                    "navigation_collision",
                    "base_plan_failed",
                    "timeout",
                ),
                elapsed_bound_s=15.0,
            )
            record(
                "navigate_unreachable_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after unreachable navigation",
            )
            record(
                "navigate_timeout_bounded",
                lambda: env.navigate_to(
                    hand="right",
                    target_xyz=[6.0, 5.0, 1.0],
                    timeout_s=0.001,
                ),
                predicate=lambda value: _bounded_failure(
                    value, stop_reasons=("timeout",)
                ),
                predicate_description="navigation timeout fails recoverably within a bound",
                expected_stop_reasons=("timeout",),
                elapsed_bound_s=5.0,
            )
            record(
                "navigate_timeout_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after navigation timeout",
            )
            for hand in HANDS:
                _place_marker_from_camera(
                    env,
                    camera="head",
                    camera_xyz=[0.0, 0.0, -1.0],
                )
                initial_observation = env.observe("head")
                if not _primitive_succeeded(initial_observation):
                    raise RuntimeError(
                        f"failed to observe private loop marker for {hand}"
                    )
                first_frame = env._frame_cache.latest("head")
                first_u, first_v = _private_current_marker_pixel(env, camera="head")
                first_projection = record(
                    f"loop_{hand}_01_pixel_to_world",
                    lambda first_frame=first_frame, first_u=first_u, first_v=first_v: (
                        env.pixel_to_world(
                            camera="head",
                            frame_id=first_frame.frame_id,
                            u=first_u,
                            v=first_v,
                        )
                    ),
                    predicate=_projection_succeeded,
                    predicate_description="loop initial public projection succeeds at <=1px",
                )
                first_xyz = np.asarray(
                    (first_projection.get("diagnostics") or {}).get("xyz")
                    if isinstance(first_projection, dict)
                    else None
                )
                navigate_call = (
                    (
                        lambda hand=hand, first_xyz=first_xyz: env.navigate_to(
                            hand=hand,
                            target_xyz=first_xyz.tolist(),
                            standoff_m=0.85,
                            timeout_s=90.0,
                        )
                    )
                    if first_xyz.shape == (3,) and np.isfinite(first_xyz).all()
                    else lambda: _explicit_failure(
                        "loop_prerequisite_failed",
                        detail="initial pixel_to_world did not produce a finite xyz",
                    )
                )
                record(
                    f"loop_{hand}_02_navigate_to",
                    navigate_call,
                    predicate=_primitive_succeeded,
                    predicate_description="loop public cuRobo BASE navigation succeeds",
                    elapsed_bound_s=95.0,
                )
                observed = record(
                    f"loop_{hand}_03_observe",
                    lambda: env.observe("head"),
                    predicate=_primitive_succeeded,
                    predicate_description="loop refreshes RGB-D after navigation",
                )
                if isinstance(observed, dict) and observed.get("frame_id"):
                    second_u, second_v = _private_current_marker_pixel(
                        env, camera="head"
                    )

                    def second_call(
                        observed: dict[str, Any] = observed,
                        second_u: int = second_u,
                        second_v: int = second_v,
                    ) -> Any:
                        return env.pixel_to_world(
                            camera="head",
                            frame_id=observed["frame_id"],
                            u=second_u,
                            v=second_v,
                        )
                else:

                    def second_call() -> Any:
                        return _explicit_failure(
                            "loop_prerequisite_failed",
                            detail="observe did not return a frame_id",
                        )

                second_projection = record(
                    f"loop_{hand}_04_pixel_to_world",
                    second_call,
                    predicate=_projection_succeeded,
                    predicate_description="loop post-navigation projection succeeds at <=1px",
                )
                xyz = np.asarray(
                    (second_projection.get("diagnostics") or {}).get("xyz")
                    if isinstance(second_projection, dict)
                    else None
                )
                if xyz.shape == (3,) and np.isfinite(xyz).all():
                    base = env._planner.backend.get_base_pose()
                    toward_base = np.r_[base[:2] - xyz[:2], 0.0]
                    toward_base /= max(float(np.linalg.norm(toward_base)), 1e-9)
                    safe = xyz + toward_base * 0.15
                    _position, quat = env._planner.backend.get_eef_pose(hand)

                    def move_call(
                        hand: str = hand,
                        safe: np.ndarray = safe,
                        quat: Any = quat,
                    ) -> Any:
                        return env.move_to(
                            hand=hand,
                            target_xyz=safe.tolist(),
                            target_quat_xyzw=np.asarray(quat).tolist(),
                            timeout_s=45.0,
                        )
                else:

                    def move_call() -> Any:
                        return _explicit_failure(
                            "loop_prerequisite_failed",
                            detail="post-navigation pixel_to_world did not produce a finite xyz",
                        )

                record(
                    f"loop_{hand}_05_move_to",
                    move_call,
                    predicate=_motion_quality_ok,
                    predicate_description=(
                        "loop public move succeeds at <=2cm/5deg for 10 holds with safe joints/dynamics"
                    ),
                )
                loop_names = [
                    f"loop_{hand}_{suffix}"
                    for suffix in (
                        "01_pixel_to_world",
                        "02_navigate_to",
                        "03_observe",
                        "04_pixel_to_world",
                        "05_move_to",
                    )
                ]
                record(
                    f"loop_{hand}_summary",
                    lambda loop_names=loop_names: {
                        "ordered_calls": loop_names,
                        "passed": [
                            bool(report["results"][name]["passed"])
                            for name in loop_names
                        ],
                    },
                    predicate=lambda value: all(value["passed"]),
                    predicate_description=(
                        "pixel_to_world -> navigate_to -> observe -> pixel_to_world -> move_to all pass"
                    ),
                )

        if args.phase in {"manipulation", "all"}:
            for hand in HANDS:
                pick_name = _PRIVATE_PICK_FIXTURES[hand][0]
                press_name = _PRIVATE_PRESS_FIXTURES[hand][0]
                pick_requested_position = np.asarray(
                    _PRIVATE_PICK_FIXTURES[hand][1], dtype=np.float64
                )
                _place_private_fixture(
                    env,
                    press_name,
                    pick_requested_position - np.array([0.0, 0.0, 0.025]),
                    settle_steps=1,
                )
                pick_center = _place_private_fixture(
                    env,
                    pick_name,
                    pick_requested_position,
                    settle_steps=5,
                )
                # ``pixel_to_world`` exposes a visible surface point, not an
                # object's privileged center pose.  Exercise pick with the
                # same public contract at the 3 cm cube's top surface.  Using
                # the center would drive the EEF 15 mm lower and make its
                # collision spheres overlap the separate support fixture.
                pick_target = pick_center + np.array(
                    [0.0, 0.0, 0.015], dtype=np.float64
                )
                # Fixture truth remains simulator-side. It is never included in
                # a public VLM observation or planner transcript.
                record(
                    f"pick_{hand}",
                    lambda hand=hand, pick_target=pick_target: env.pick(
                        hand=hand,
                        target_xyz=pick_target.tolist(),
                        approach_vector=[0.0, 0.0, -1.0],
                        timeout_s=90.0,
                    ),
                    predicate=_primitive_succeeded,
                    predicate_description="real public pick succeeds with attached collision body",
                    elapsed_bound_s=100.0,
                )
                record(
                    f"release_{hand}_after_pick",
                    lambda hand=hand: env.release(
                        hand=hand,
                        opening=1.0,
                        retreat_m=0.0,
                        timeout_s=30.0,
                    ),
                    predicate=_primitive_succeeded,
                    predicate_description="real public release succeeds and detaches held body",
                    elapsed_bound_s=35.0,
                )
                _deactivate_private_fixture(env, pick_name)
                _deactivate_private_fixture(env, press_name)
                press_position = _place_private_fixture(
                    env,
                    press_name,
                    _PRIVATE_PRESS_FIXTURES[hand][1],
                    settle_steps=1,
                )
                press_target = press_position + np.array(
                    [0.0, 0.0, 0.01], dtype=np.float64
                )
                record(
                    f"press_{hand}",
                    lambda hand=hand, press_target=press_target: env.press(
                        hand=hand,
                        target_xyz=press_target.tolist(),
                        press_direction=[0.0, 0.0, -1.0],
                        timeout_s=60.0,
                    ),
                    predicate=_primitive_succeeded,
                    predicate_description="real guarded public press succeeds",
                    elapsed_bound_s=70.0,
                )
                _deactivate_private_fixture(env, press_name)
                record(
                    f"manipulation_{hand}_health_observe",
                    lambda: env.observe("head"),
                    predicate=_primitive_succeeded,
                    predicate_description="RPC remains healthy after manipulation sequence",
                )

        if args.phase in {"adversarial", "all"}:
            collision_origin, collision_quat = env._planner.backend.get_eef_pose("left")
            collision_target = np.asarray(
                collision_origin, dtype=np.float64
            ) + np.array([0.08, 0.0, 0.0], dtype=np.float64)
            collision_precheck = record(
                "collision_free_precheck",
                lambda: env.move_to(
                    hand="left",
                    target_xyz=collision_target.tolist(),
                    target_quat_xyzw=np.asarray(collision_quat).tolist(),
                    plan_only=True,
                    timeout_s=15.0,
                ),
                predicate=_primitive_succeeded,
                predicate_description="same collision target is reachable before obstacle injection",
                elapsed_bound_s=20.0,
            )
            obstacle = env._env.omnigibson_env.scene.object_registry(
                "name", _PRIVATE_COLLISION_FIXTURE
            )
            if obstacle is None:
                raise RuntimeError("private collision obstacle was not loaded")
            obstacle.set_position_orientation(
                position=collision_target,
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
            import omnigibson as og

            for _ in range(3):
                og.sim.render()
            record(
                "collision_bounded_failure",
                lambda: env.move_to(
                    hand="left",
                    target_xyz=collision_target.tolist(),
                    target_quat_xyzw=np.asarray(collision_quat).tolist(),
                    timeout_s=15.0,
                ),
                predicate=lambda value: (
                    _bounded_failure(
                        value,
                        stop_reasons=(
                            "unreachable",
                            "trajectory_collision",
                            "unexpected_collision",
                        ),
                    )
                    and _primitive_succeeded(collision_precheck)
                ),
                predicate_description=(
                    "same real target plans when clear and is boundedly rejected after a fixed obstacle is inserted"
                ),
                expected_stop_reasons=(
                    "unreachable",
                    "trajectory_collision",
                    "unexpected_collision",
                ),
                elapsed_bound_s=25.0,
            )
            obstacle.set_position_orientation(
                position=[100.0, 100.0, 100.0],
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
            for _ in range(3):
                og.sim.render()
            record(
                "collision_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after collision rejection",
            )

            def joint_limit_call() -> Any:
                robot = env._robot()
                if robot is None:
                    raise RuntimeError(
                        "R1Pro unavailable for joint-limit fault injection"
                    )
                q = _as_numpy(robot.get_joint_positions()).reshape(-1)
                index = int(_as_numpy(robot.arm_control_idx["right"]).reshape(-1)[-1])
                lower, upper = robot.control_limits["position"]
                old_lower = lower[index].clone()
                old_upper = upper[index].clone()
                report["harness_configuration"]["joint_limit_fault_injection"] = {
                    "joint": str(robot.dof_names_ordered[index]),
                    "current": float(q[index]),
                    "injected_lower": float(q[index] - 1.0),
                    "injected_upper": float(q[index] + 0.01),
                    "purpose": "exercise the production joint-margin abort before motion",
                }
                try:
                    lower[index] = float(q[index] - 1.0)
                    upper[index] = float(q[index] + 0.01)
                    position, quat = env._planner.backend.get_eef_pose("right")
                    return env.move_to(
                        hand="right",
                        target_xyz=(
                            np.asarray(position, dtype=np.float64)
                            + np.array([0.01, 0.0, 0.0])
                        ).tolist(),
                        target_quat_xyzw=np.asarray(quat).tolist(),
                        timeout_s=15.0,
                    )
                finally:
                    lower[index].copy_(old_lower)
                    upper[index].copy_(old_upper)

            record(
                "joint_limit_bounded_failure",
                joint_limit_call,
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=("joint_limit_margin",),
                ),
                predicate_description="production move monitor aborts on an injected 0.01rad real-joint margin",
                expected_stop_reasons=("joint_limit_margin",),
                elapsed_bound_s=25.0,
            )
            record(
                "joint_limit_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after joint-limit rejection",
            )

            def in_flight_planner_timeout_call() -> Any:
                generator = env._planner.backend._generator(kind="arm", hand="left")
                original = generator.compute_trajectories
                report["harness_configuration"]["planner_timeout_fault_injection"] = {
                    "kind": "synchronous_compute_delay",
                    "delay_s": 1.0,
                    "public_timeout_s": 0.05,
                    "purpose": (
                        "exercise the production wall-clock deadline while a "
                        "cuRobo entrypoint is in flight"
                    ),
                }

                def delayed_compute(*call_args: Any, **call_kwargs: Any) -> Any:
                    time.sleep(1.0)
                    return original(*call_args, **call_kwargs)

                generator.compute_trajectories = delayed_compute
                position, quat = env._planner.backend.get_eef_pose("left")
                try:
                    return env.move_to(
                        hand="left",
                        target_xyz=(
                            np.asarray(position, dtype=np.float64)
                            + np.array([0.01, 0.0, 0.0])
                        ).tolist(),
                        target_quat_xyzw=np.asarray(quat).tolist(),
                        timeout_s=0.05,
                    )
                finally:
                    generator.compute_trajectories = original

            record(
                "in_flight_planner_timeout_bounded_failure",
                in_flight_planner_timeout_call,
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=("timeout",),
                ),
                predicate_description=(
                    "an in-flight synchronous planner call times out within its "
                    "public wall-clock bound"
                ),
                expected_stop_reasons=("timeout",),
                elapsed_bound_s=2.0,
            )
            record(
                "in_flight_planner_timeout_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description=(
                    "RPC remains healthy after an in-flight planner timeout"
                ),
            )

            def recovery_plan_call() -> Any:
                position, quat = env._planner.backend.get_eef_pose("left")
                return env.move_to(
                    hand="left",
                    target_xyz=np.asarray(position).tolist(),
                    target_quat_xyzw=np.asarray(quat).tolist(),
                    plan_only=True,
                    timeout_s=120.0,
                )

            record(
                "in_flight_timeout_generator_rebuild_and_warmup",
                recovery_plan_call,
                predicate=_primitive_succeeded,
                predicate_description=(
                    "the quarantined generator is freshly rebuilt, collision-warmed, "
                    "and completes a real cuRobo plan before reuse"
                ),
                elapsed_bound_s=180.0,
            )

            record(
                "native_curobo_timeout_bounded_failure",
                lambda: env.move_to(
                    hand="left",
                    target_xyz=(
                        np.asarray(
                            env._planner.backend.get_eef_pose("left")[0],
                            dtype=np.float64,
                        )
                        + np.array([0.01, 0.0, 0.0])
                    ).tolist(),
                    timeout_s=0.001,
                ),
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=("timeout",),
                ),
                predicate_description=(
                    "a real cuRobo compute call is interrupted and quarantined at "
                    "the native planner boundary"
                ),
                expected_stop_reasons=("timeout",),
                elapsed_bound_s=2.0,
            )
            record(
                "native_curobo_timeout_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after native cuRobo timeout",
            )
            record(
                "native_timeout_generator_rebuild_and_warmup",
                recovery_plan_call,
                predicate=_primitive_succeeded,
                predicate_description=(
                    "native-timeout generator is rebuilt and real-planned before reuse"
                ),
                elapsed_bound_s=180.0,
            )

            def stall_call() -> Any:
                robot = env._robot()
                if robot is None:
                    raise RuntimeError(
                        "R1Pro unavailable for controller stall injection"
                    )
                controlled = list(_as_numpy(robot.trunk_control_idx).reshape(-1))
                controlled.extend(
                    _as_numpy(robot.arm_control_idx["left"]).reshape(-1).tolist()
                )
                joints = [
                    robot.joints[robot.dof_names_ordered[int(i)]] for i in controlled
                ]
                stiffness = [joint.stiffness.clone() for joint in joints]
                report["harness_configuration"]["stall_fault_injection"] = {
                    "joints": [
                        str(robot.dof_names_ordered[int(i)]) for i in controlled
                    ],
                    "stiffness": "temporarily_zero",
                    "max_no_improvement_steps": int(env._planner.max_stall_steps),
                    "purpose": "exercise production stalled_tracking on the live articulation",
                }
                position, quat = env._planner.backend.get_eef_pose("left")
                try:
                    for joint in joints:
                        joint.stiffness = 0.0
                    return env.move_to(
                        hand="left",
                        target_xyz=(
                            np.asarray(position, dtype=np.float64)
                            + np.array([0.06, 0.0, 0.0])
                        ).tolist(),
                        target_quat_xyzw=np.asarray(quat).tolist(),
                        timeout_s=25.0,
                    )
                finally:
                    for joint, original in zip(joints, stiffness):
                        joint.stiffness = float(original)

            record(
                "stall_bounded_failure",
                stall_call,
                predicate=lambda value: _bounded_failure(
                    value,
                    stop_reasons=("stalled_tracking",),
                ),
                predicate_description="real public action returns stalled_tracking within 20 no-progress steps",
                expected_stop_reasons=("stalled_tracking",),
                elapsed_bound_s=30.0,
            )
            record(
                "stall_health_observe",
                lambda: env.observe("head"),
                predicate=_primitive_succeeded,
                predicate_description="RPC remains healthy after stall fixture result",
            )

        if args.phase in {"calibration", "all"}:

            def calibrate_markers() -> dict[str, Any]:
                if args.marker_calibration_json:
                    return _explicit_failure(
                        "external_marker_fixture_forbidden",
                        detail=(
                            "final calibration acceptance requires marker truth captured "
                            "from this live simulator process"
                        ),
                    )
                payload = _capture_live_marker_fixture(env)
                marker_path = output_dir / "private_live_marker_fixture.json"
                marker_bytes = json.dumps(_artifact_jsonable(payload), indent=2).encode(
                    "utf-8"
                )
                marker_path.write_bytes(marker_bytes)
                marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
                cameras = payload.get("cameras")
                if not isinstance(cameras, dict):
                    return _explicit_failure(
                        "invalid_marker_fixture",
                        detail="marker JSON must contain a cameras mapping",
                    )
                profiles: dict[str, Any] = {}
                provenance: dict[str, Any] = {}
                for camera in CAMERAS:
                    fixture = cameras.get(camera)
                    if not isinstance(fixture, dict):
                        return _explicit_failure(
                            "invalid_marker_fixture",
                            detail=f"missing real marker correspondences for {camera}",
                        )
                    train = fixture.get("train_samples")
                    heldout = fixture.get("heldout_samples")
                    if not isinstance(train, list) or len(train) < 3:
                        return _explicit_failure(
                            "invalid_marker_fixture",
                            detail=f"{camera} requires at least 3 train marker samples",
                        )
                    if not isinstance(heldout, list) or len(heldout) < 3:
                        return _explicit_failure(
                            "invalid_marker_fixture",
                            detail=f"{camera} requires at least 3 heldout marker samples",
                        )
                    train_raw = [sample["raw_camera_xyz"] for sample in train]
                    train_truth = [sample["true_camera_xyz"] for sample in train]
                    if _point_set_rank(train_raw) < 2:
                        return _explicit_failure(
                            "invalid_marker_fixture",
                            detail=f"{camera} train marker points are rank-deficient",
                        )
                    train_pair_errors = np.linalg.norm(
                        np.asarray(train_raw, dtype=np.float64)
                        - np.asarray(train_truth, dtype=np.float64),
                        axis=1,
                    )
                    if (
                        float(np.max(train_pair_errors))
                        > PRIVATE_MARKER_DEPTH_CONSISTENCY_M
                    ):
                        return _explicit_failure(
                            "invalid_marker_fixture",
                            detail=f"{camera} train marker RGB-D is not pose-synchronous",
                        )
                    raw = np.asarray(
                        [sample["raw_camera_xyz"] for sample in heldout],
                        dtype=np.float64,
                    )
                    truth = np.asarray(
                        [sample["true_camera_xyz"] for sample in heldout],
                        dtype=np.float64,
                    )
                    raw_errors = np.linalg.norm(raw - truth, axis=1)
                    profile = fit_camera_correction_profile(
                        camera=camera,
                        train_samples=train,
                        heldout_samples=heldout,
                    )
                    encoded = correction_profile_to_json(profile)
                    corrected = np.stack(
                        [profile.apply_camera_point(point) for point in raw],
                        axis=0,
                    )
                    corrected_errors = np.linalg.norm(corrected - truth, axis=1)
                    encoded["heldout_raw_median_m"] = float(np.median(raw_errors))
                    encoded["heldout_raw_p95_m"] = float(np.percentile(raw_errors, 95))
                    encoded["heldout_raw_max_m"] = float(np.max(raw_errors))
                    encoded["heldout_corrected_max_m"] = float(np.max(corrected_errors))
                    profiles[camera] = encoded
                    provenance[camera] = {
                        "train_count": len(train),
                        "heldout_count": len(heldout),
                        "train_affine_rank": _point_set_rank(train_raw),
                        "max_train_pair_error_m": float(np.max(train_pair_errors)),
                        "frame_ids": [
                            sample.get("frame_id") for sample in [*train, *heldout]
                        ],
                        "capture_group_ids": [
                            sample.get("capture_group") for sample in [*train, *heldout]
                        ],
                        "candidate_indices": [
                            sample.get("candidate_index")
                            for sample in [*train, *heldout]
                        ],
                    }
                return {
                    "primitive_success": True,
                    "task_success": False,
                    "stop_reason": "marker_calibration_evaluated",
                    "recoverable": True,
                    "profiles": profiles,
                    "fixture_source": payload.get("source"),
                    "fixture_path": str(marker_path),
                    "fixture_sha256": marker_sha256,
                    "provenance": provenance,
                }

            def calibration_passed(value: Any) -> bool:
                if not _primitive_succeeded(value):
                    return False
                profiles = value.get("profiles")
                if not isinstance(profiles, dict) or set(profiles) != set(CAMERAS):
                    return False
                if value.get("fixture_source") != "live_simulator_marker":
                    return False
                fixture_sha256 = value.get("fixture_sha256")
                if not isinstance(fixture_sha256, str) or len(fixture_sha256) != 64:
                    return False
                provenance = value.get("provenance")
                if not isinstance(provenance, dict) or set(provenance) != set(CAMERAS):
                    return False
                for camera_provenance in provenance.values():
                    if not (
                        camera_provenance.get("train_count") >= 3
                        and camera_provenance.get("heldout_count") >= 3
                        and camera_provenance.get("train_affine_rank") >= 2
                        and float(
                            camera_provenance.get("max_train_pair_error_m", math.inf)
                        )
                        <= PRIVATE_MARKER_DEPTH_CONSISTENCY_M
                        and len(set(camera_provenance.get("frame_ids", ()))) >= 6
                        and len(set(camera_provenance.get("capture_group_ids", ())))
                        >= 6
                    ):
                        return False
                for profile in profiles.values():
                    metrics = profile.get("metrics") or {}
                    enabled = profile.get("enabled") is True
                    try:
                        raw_median = float(profile["heldout_raw_median_m"])
                        raw_p95 = float(profile["heldout_raw_p95_m"])
                        raw_max = float(profile["heldout_raw_max_m"])
                        corrected_max = float(profile["heldout_corrected_max_m"])
                        after_median = float(metrics["after_median_m"])
                        before_median = float(metrics["before_median_m"])
                        after_p95 = float(metrics["after_p95_m"])
                        before_p95 = float(metrics["before_p95_m"])
                    except (KeyError, TypeError, ValueError):
                        return False
                    if enabled:
                        if not (
                            after_median <= 0.02
                            and after_p95 <= 0.02
                            and corrected_max <= 0.02
                            and after_median <= before_median * 0.8
                            and after_p95 <= before_p95
                        ):
                            return False
                    elif not (
                        np.allclose(profile["raw_to_corrected_camera"], np.eye(4))
                        and raw_median <= 0.02
                        and raw_p95 <= 0.02
                        and raw_max <= 0.02
                    ):
                        return False
                return True

            record(
                "heldout_marker_calibration",
                calibrate_markers,
                predicate=calibration_passed,
                predicate_description=(
                    "live simulator-marker heldout error is <=2cm; correction is enabled "
                    "only with >=20% median improvement and non-worse p95, otherwise identity remains"
                ),
            )

        report["status"] = "failed" if report["mandatory_failures"] else "passed"
        report["env_steps"] = env._env_steps
        report["official_task_success"] = bool(
            isinstance(env._last_info, dict)
            and isinstance(env._last_info.get("done"), dict)
            and env._last_info["done"].get("success", False)
        )
        report["acceptance"] = {
            "passed": report["status"] == "passed",
            "mandatory_failure_count": len(report["mandatory_failures"]),
            "mandatory_failures": list(report["mandatory_failures"]),
        }
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["status"] = "failed"
                report["close_error"] = f"{type(exc).__name__}: {exc}"
                if "environment_close" not in report["mandatory_failures"]:
                    report["mandatory_failures"].append("environment_close")
                save()

    if report.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
