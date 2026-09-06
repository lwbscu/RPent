# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometry helpers for the RPent YAM facade.

The motion runtime and IK implementation remain owned by RLinf/i2rt. This
module only normalizes qpos14, calibration transforms, and planning inputs for
the RPent agent boundary.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robots.yam import contracts as _contracts

CAMERA_NAMES = tuple(getattr(_contracts, "CAMERA_NAMES", _contracts.YAM_CAMERA_NAMES))
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
ARM_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.array([6, 13])
DEFAULT_JOINT_LIMIT_MIN = np.array(
    [
        [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
        [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
    ],
    dtype=np.float64,
)
DEFAULT_JOINT_LIMIT_MAX = np.array(
    [
        [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
        [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
    ],
    dtype=np.float64,
)


def bootstrap_rlinf_root(config: dict[str, Any] | None = None) -> Path | None:
    """Add the configured RLinf checkout to ``sys.path`` if one is provided."""
    config = config or {}
    env_names = [
        str(config.get("rlinf_root_env", "RPENT_RLINF_ROOT")),
        "RPENT_RLINF_ROOT",
        "RLINF_REPO_PATH",
        "YAM_RLINF_ROOT",
    ]
    root_value = config.get("rlinf_root") or next(
        (os.environ[name] for name in env_names if os.environ.get(name)),
        None,
    )
    if root_value is None:
        return None
    root = Path(root_value).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"configured RLinf root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def import_rlinf_object(
    module_name: str, object_name: str, config: dict[str, Any] | None = None
) -> Any:
    """Import one RLinf object with a dependency-focused error message."""
    bootstrap_rlinf_root(config)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = error.name or module_name
        raise RuntimeError(
            f"missing dependency while importing {module_name}: {missing}. "
            "Set RPENT_RLINF_ROOT, RLINF_REPO_PATH, YAM_RLINF_ROOT, or "
            "config['rlinf_root'] to the RLinf checkout and install the pinned "
            "YAM/i2rt dependencies on the robot host."
        ) from error
    return getattr(module, object_name)


def as_qpos14(value: Any, *, name: str = "qpos") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (14,):
        raise ValueError(f"{name} must have shape (14,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array[GRIPPER_INDICES] < 0.0) | (array[GRIPPER_INDICES] > 1.0)):
        raise ValueError(f"{name} grippers must be in [0,1], 0=closed, 1=open")
    return array.copy()


def split_qpos14(qpos: Any) -> tuple[np.ndarray, np.ndarray]:
    vector = as_qpos14(qpos)
    return vector[ARM_SLICES["left"]].copy(), vector[ARM_SLICES["right"]].copy()


def pack_qpos14(left: Any, right: Any) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != (7,) or right_array.shape != (7,):
        raise ValueError(
            f"left/right YAM qpos must be 7D, got {left_array.shape}/{right_array.shape}"
        )
    return as_qpos14(np.concatenate([left_array, right_array]))


def joint_limits_from_config(
    config: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or {}
    lower = np.asarray(
        config.get("joint_limit_min", DEFAULT_JOINT_LIMIT_MIN), dtype=np.float64
    )
    upper = np.asarray(
        config.get("joint_limit_max", DEFAULT_JOINT_LIMIT_MAX), dtype=np.float64
    )
    if lower.shape != (2, 6) or upper.shape != (2, 6):
        raise ValueError(
            f"YAM joint limits must have shape (2,6), got {lower.shape}/{upper.shape}"
        )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("YAM joint limits must be finite")
    if not np.all(lower < upper):
        raise ValueError("each YAM joint lower limit must be below its upper limit")
    return lower.copy(), upper.copy()


def enforce_hard_limits(
    qpos: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str = "qpos",
) -> np.ndarray:
    vector = as_qpos14(qpos, name=name)
    left, right = split_qpos14(vector)
    for arm_name, arm, lo, hi in (
        ("left", left, lower[0], upper[0]),
        ("right", right, lower[1], upper[1]),
    ):
        below = arm[:6] < lo
        above = arm[:6] > hi
        if np.any(below) or np.any(above):
            indexes = np.flatnonzero(below | above).tolist()
            raise ValueError(
                f"{name} {arm_name} arm joints outside hard limits at {indexes}"
            )
    return vector


def apply_previous_command_slew(
    target: Any,
    previous_command: Any,
    max_joint_delta: float | None,
) -> tuple[np.ndarray, bool]:
    """Clamp arm joints relative to the previous command, not measured qpos."""
    accepted = as_qpos14(target, name="target")
    if max_joint_delta is None:
        return accepted, False
    delta = float(max_joint_delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("max_joint_delta_per_step must be finite and positive")
    previous = as_qpos14(previous_command, name="previous_command")
    before = accepted.copy()
    accepted[ARM_JOINT_INDICES] = np.clip(
        accepted[ARM_JOINT_INDICES],
        previous[ARM_JOINT_INDICES] - delta,
        previous[ARM_JOINT_INDICES] + delta,
    )
    return accepted, not np.array_equal(before, accepted)


def matrix4(value: Any, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} bottom row must be [0,0,0,1]")
    return matrix.copy()


def pose_to_matrix(value: Any) -> np.ndarray:
    """Accept either 4x4 or xyz+wxyz pose and return a 4x4 matrix."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        return matrix4(array, name="target_pose")
    if array.shape != (7,):
        raise ValueError("target_pose must be 4x4 or xyz+wxyz shape (7,)")
    xyz = array[:3]
    quat = array[3:]
    if not np.isfinite(array).all():
        raise ValueError("target_pose must contain only finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        raise ValueError("target_pose quaternion must be non-zero")
    w, x, y, z = quat / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rot
    result[:3, 3] = xyz
    return result


def matrix_to_xyz_wxyz(matrix: Any) -> np.ndarray:
    transform = matrix4(matrix)
    rot = transform[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rot)))
        if index == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return np.concatenate([transform[:3, 3], quat])


def deproject_pixel(u: float, v: float, depth_m: Any, intrinsic_K: Any) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    k_matrix = np.asarray(intrinsic_K, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth image must be 2D meters")
    if k_matrix.shape != (3, 3):
        raise ValueError("intrinsic_K must be 3x3")
    px = int(round(u))
    py = int(round(v))
    patch = depth[max(0, py - 3) : py + 4, max(0, px - 3) : px + 4]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if valid.size == 0:
        raise ValueError(f"no valid metric depth near pixel ({px}, {py})")
    z = float(np.median(valid))
    return np.array(
        [
            (px - k_matrix[0, 2]) * z / k_matrix[0, 0],
            (py - k_matrix[1, 2]) * z / k_matrix[1, 1],
            z,
        ],
        dtype=np.float64,
    )


def hover_target_from_pixel(
    pixel_base: Any, axis_base: Any, hover_m: float
) -> np.ndarray:
    """Build the same top-down hover target convention used by RLinf tooling."""
    point = np.asarray(pixel_base, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_base, dtype=np.float64).reshape(3)
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_xy = axis - np.dot(axis, z_up) * z_up
    if np.linalg.norm(axis_xy) < 1e-6:
        axis_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis_xy /= np.linalg.norm(axis_xy)
    y_site = np.cross(axis_xy, z_up)
    y_site /= np.linalg.norm(y_site)
    x_site = z_up - np.dot(z_up, y_site) * y_site
    x_site /= np.linalg.norm(x_site)
    z_site = np.cross(x_site, y_site)
    target = np.eye(4, dtype=np.float64)
    target[:3, 0] = x_site
    target[:3, 1] = y_site
    target[:3, 2] = z_site
    target[:3, 3] = point + np.array([0.0, 0.0, float(hover_m)])
    return target


def _load_json_path(value: str | os.PathLike[str] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return json.loads(Path(value).expanduser().read_text())


@dataclass(frozen=True)
class YamCalibration:
    """Transforms into the RPent world frame, defined as the left robot base."""

    world_frame: str
    camera_to_world: dict[str, np.ndarray]
    intrinsic_K: dict[str, np.ndarray]
    world_from_right_base: np.ndarray | None
    wrist_grasp_from_camera: dict[str, np.ndarray]
    notes: tuple[str, ...] = ()

    def cam2world_cv(self, camera_name: str) -> np.ndarray:
        if camera_name not in self.camera_to_world:
            raise ValueError(f"missing cam2world_cv for YAM camera {camera_name!r}")
        return self.camera_to_world[camera_name].copy()

    def arm_target_from_world(
        self, arm: Literal["left", "right"], target_world: Any
    ) -> np.ndarray:
        target = pose_to_matrix(target_world)
        if arm == "left":
            return target
        if self.world_from_right_base is None:
            raise ValueError(
                "right-arm planning requires T_world_from_right_base; provide it "
                "explicitly or derive it from top camera calibration"
            )
        return np.linalg.inv(self.world_from_right_base) @ target

    def pose_to_world(
        self, arm: Literal["left", "right"], pose_in_arm_base: Any
    ) -> np.ndarray:
        pose = matrix4(pose_in_arm_base, name=f"{arm}_eef_pose")
        if arm == "left":
            return pose
        if self.world_from_right_base is None:
            raise ValueError("right-arm pose requires T_world_from_right_base")
        return self.world_from_right_base @ pose


def load_calibration(config: dict[str, Any] | None) -> YamCalibration:
    """Load RPent YAM calibration.

    Accepted keys are intentionally explicit. ``cam2world_cv`` means
    ``T_left_base_from_camera_cv`` because the RPent world frame is left_base.
    Legacy RLinf calibration names are interpreted from the producer's matrix
    equations: both ``T_base_to_cam`` and ``T_grasp_to_cam`` map camera points
    into the named robot frame, despite their historical spelling.
    """
    config = config or {}
    calibration = dict(config.get("calibration", {}))
    calibration.update(_load_json_path(config.get("extrinsics_path")))
    world_frame = str(calibration.get("world_frame", "left_base"))
    if world_frame != "left_base":
        raise ValueError("YAM RPent world_frame must be 'left_base'")

    notes: list[str] = []
    cameras_cfg = dict(calibration.get("cameras", {}))
    camera_to_world: dict[str, np.ndarray] = {}
    intrinsic_k: dict[str, np.ndarray] = {}
    right_from_camera: dict[str, np.ndarray] = {}
    wrist_grasp_from_camera: dict[str, np.ndarray] = {}

    def _record_wrist_transform(name: str, values: dict[str, Any], prefix: str) -> None:
        if name not in {"left", "right"}:
            return
        if values.get("T_grasp_from_camera") is not None:
            wrist_grasp_from_camera[name] = matrix4(
                values["T_grasp_from_camera"], name=f"{prefix}.T_grasp_from_camera"
            )
        elif values.get("T_eef_from_camera") is not None:
            wrist_grasp_from_camera[name] = matrix4(
                values["T_eef_from_camera"], name=f"{prefix}.T_eef_from_camera"
            )
        elif values.get("T_grasp_to_cam") is not None:
            # RLinf solve_handeye.py verifies FK @ t_tcp_cam @ board_in_cam
            # and writes t_tcp_cam under this historical key. Do not invert.
            wrist_grasp_from_camera[name] = matrix4(
                values["T_grasp_to_cam"], name=f"{prefix}.T_grasp_to_cam"
            )

    for name, spec in cameras_cfg.items():
        values = dict(spec)
        if values.get("intrinsic_K") is not None:
            intrinsic_k[str(name)] = np.asarray(values["intrinsic_K"], dtype=np.float64)
        if values.get("cam2world_cv") is not None:
            camera_to_world[str(name)] = matrix4(
                values["cam2world_cv"], name=f"{name}.cam2world_cv"
            )
        if values.get("T_left_base_from_camera") is not None:
            camera_to_world[str(name)] = matrix4(
                values["T_left_base_from_camera"],
                name=f"{name}.T_left_base_from_camera",
            )
        if values.get("T_right_base_from_camera") is not None:
            right_from_camera[str(name)] = matrix4(
                values["T_right_base_from_camera"],
                name=f"{name}.T_right_base_from_camera",
            )
        _record_wrist_transform(str(name), values, f"cameras.{name}")

    for wrist_key, arm_name in (("left_wrist", "left"), ("right_wrist", "right")):
        wrist_values = calibration.get(wrist_key)
        if isinstance(wrist_values, dict):
            _record_wrist_transform(arm_name, wrist_values, wrist_key)

    for arm_name, wrist_values in dict(config.get("wrist_cameras", {})).items():
        if isinstance(wrist_values, dict):
            _record_wrist_transform(
                str(arm_name), wrist_values, f"wrist_cameras.{arm_name}"
            )

    legacy_top = calibration.get("top_camera", {}).get("T_base_to_cam", {})
    if legacy_top:
        if "via_left_arm" in legacy_top:
            camera_to_world["top"] = matrix4(
                legacy_top["via_left_arm"],
                name="top_camera.T_base_to_cam.via_left_arm",
            )
        if "via_right_arm" in legacy_top:
            right_from_camera["top"] = matrix4(
                legacy_top["via_right_arm"],
                name="top_camera.T_base_to_cam.via_right_arm",
            )
        notes.append("legacy_top_T_base_to_cam_used_as_base_from_camera")

    world_from_right = None
    if calibration.get("T_left_base_from_right_base") is not None:
        world_from_right = matrix4(
            calibration["T_left_base_from_right_base"],
            name="T_left_base_from_right_base",
        )
    elif "top" in camera_to_world and "top" in right_from_camera:
        world_from_right = camera_to_world["top"] @ np.linalg.inv(
            right_from_camera["top"]
        )
        notes.append("T_left_base_from_right_base_derived_from_top_camera")

    if world_from_right is not None:
        for name, transform in right_from_camera.items():
            camera_to_world.setdefault(name, world_from_right @ transform)

    intrinsics_path = config.get("intrinsics_path")
    intrinsics = _load_json_path(intrinsics_path)
    if intrinsics:
        if "cameras" in intrinsics:
            for name, spec in intrinsics["cameras"].items():
                intrinsic_k[str(name)] = np.asarray(
                    spec["intrinsic_K"], dtype=np.float64
                )
        elif "fx" in intrinsics:
            intrinsic_k.setdefault(
                "top",
                np.array(
                    [
                        [intrinsics["fx"], 0.0, intrinsics["cx"]],
                        [0.0, intrinsics["fy"], intrinsics["cy"]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            )

    for name, k_matrix in list(intrinsic_k.items()):
        if k_matrix.shape != (3, 3) or not np.isfinite(k_matrix).all():
            raise ValueError(f"{name}.intrinsic_K must be finite 3x3")

    return YamCalibration(
        world_frame=world_frame,
        camera_to_world=camera_to_world,
        intrinsic_K=intrinsic_k,
        world_from_right_base=world_from_right,
        wrist_grasp_from_camera=wrist_grasp_from_camera,
        notes=tuple(notes),
    )


class YamGeometry:
    """Planner and pose adapter backed by RLinf's YAM kinematics."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        kinematics: dict[str, Any] | Any | None = None,
    ) -> None:
        self.config = config or {}
        self.calibration = load_calibration(self.config)
        self.lower, self.upper = joint_limits_from_config(self.config)
        self.table_z = (
            None
            if self.config.get("table_z") is None
            else float(self.config["table_z"])
        )
        self.table_clearance_m = float(self.config.get("table_clearance_m", 0.03))
        self.path_joint_delta = float(self.config.get("path_joint_delta", 0.02))
        self._kinematics = kinematics

    def _kinematics_for(self, arm: Literal["left", "right"]) -> Any:
        if isinstance(self._kinematics, dict):
            if arm in self._kinematics:
                return self._kinematics[arm]
        elif self._kinematics is not None:
            return self._kinematics
        adapter_cls = import_rlinf_object(
            "rlinf.envs.realworld.yam.kinematics",
            "YamKinematicsAdapter",
            self.config,
        )
        adapter = adapter_cls(
            joint_lower=self.lower[0 if arm == "left" else 1],
            joint_upper=self.upper[0 if arm == "left" else 1],
        )
        if self._kinematics is None:
            self._kinematics = {}
        self._kinematics[arm] = adapter
        return adapter

    def eef_pose(self, arm: Literal["left", "right"], qpos14: Any) -> np.ndarray:
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        arm_qpos = qpos[ARM_SLICES[arm]]
        fk = self._kinematics_for(arm).fk(arm_qpos[:6], float(arm_qpos[6]))
        return matrix_to_xyz_wxyz(self.calibration.pose_to_world(arm, fk))

    def robot_state(self, qpos14: Any) -> dict[str, Any]:
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        return {
            "qpos": qpos.astype(np.float64),
            "left_eef_pose": self.eef_pose("left", qpos),
            "right_eef_pose": self.eef_pose("right", qpos),
            "world_frame": self.calibration.world_frame,
        }

    def camera_meta_for_qpos(
        self, camera_name: str, base_meta: dict[str, Any], qpos14: Any
    ) -> dict[str, Any]:
        """Return camera metadata with wrist cam poses tied to measured qpos."""
        name = str(camera_name)
        meta = dict(base_meta)
        if name == "top":
            meta["cam2world_cv"] = self.calibration.cam2world_cv("top")
            meta["cam2world_source"] = "static_top_calibration"
            return meta
        if name not in {"left", "right"}:
            raise ValueError(f"unknown YAM camera {camera_name!r}")
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        arm_qpos = qpos[ARM_SLICES[name]]
        grasp_from_camera = self.calibration.wrist_grasp_from_camera.get(name)
        if grasp_from_camera is None:
            raise ValueError(
                f"{name} wrist camera requires dynamic hand-eye calibration: "
                "provide T_grasp_from_camera/T_eef_from_camera or RLinf "
                f"{name}_wrist.T_grasp_to_cam"
            )
        base_from_grasp = self._kinematics_for(name).fk(
            arm_qpos[:6], float(arm_qpos[6])
        )
        world_from_camera = self.calibration.pose_to_world(
            name, base_from_grasp
        ) @ matrix4(grasp_from_camera, name=f"{name}.T_grasp_from_camera")
        meta["cam2world_cv"] = world_from_camera
        meta["cam2world_source"] = "dynamic_fk_current_qpos_approx_cached_frame"
        meta["cam2world_limitation"] = (
            "computed from current measured qpos for a cached RGBD frame; use only "
            "when robot is stationary or inspect frame_age_s/qpos_delta metadata"
        )
        meta["qpos_used_for_cam2world"] = qpos.copy()
        return meta

    def plan_arm_path(
        self,
        arm: Literal["left", "right"],
        target_pose: Any,
        *,
        current_qpos14: Any,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        qpos = enforce_hard_limits(current_qpos14, self.lower, self.upper)
        arm_index = 0 if arm == "left" else 1
        seed = qpos[ARM_SLICES[arm]][:6]
        gripper = float(qpos[ARM_SLICES[arm]][6])
        target_in_base = self.calibration.arm_target_from_world(arm, target_pose)
        kin = self._kinematics_for(arm)
        result = kin.solve(target_in_base, seed, gripper)
        if not bool(result.success):
            return {
                "status": "Failure",
                "position": None,
                "reason": getattr(result, "reason", "ik_failed"),
                "position_error": float(getattr(result, "position_error", np.inf)),
                "rotation_error": float(getattr(result, "rotation_error", np.inf)),
            }

        q_target = np.asarray(result.q_target, dtype=np.float64).reshape(6)
        if np.any(q_target < self.lower[arm_index]) or np.any(
            q_target > self.upper[arm_index]
        ):
            return {
                "status": "Failure",
                "position": None,
                "reason": "joint_limits",
            }
        max_delta = float(np.max(np.abs(q_target - seed)))
        steps = max(2, int(np.ceil(max_delta / self.path_joint_delta)) + 1)
        position = np.linspace(seed, q_target, steps, dtype=np.float64)
        table_check = self._check_table_guard(arm, position, gripper, kin)
        if not table_check["ok"]:
            return {
                "status": "Failure",
                "position": None,
                "reason": "table_guard",
                **table_check,
            }
        return {
            "status": "Success",
            "position": position,
            "reason": None,
            "position_error": float(result.position_error),
            "rotation_error": float(result.rotation_error),
            "table_guard": table_check,
        }

    def check_qpos_transition(
        self, previous_qpos14: Any, target_qpos14: Any, *, samples: int = 8
    ) -> dict[str, Any]:
        """Check TCP height along the commanded qpos segment."""
        if self.table_z is None:
            return {
                "ok": False,
                "checked": False,
                "reason": "table_not_configured",
                "limitation": "table_z is required before real qpos motion",
            }
        previous = enforce_hard_limits(previous_qpos14, self.lower, self.upper)
        target = enforce_hard_limits(target_qpos14, self.lower, self.upper)
        min_clearance = float("inf")
        for alpha in np.linspace(0.0, 1.0, max(2, int(samples))):
            qpos = previous + alpha * (target - previous)
            for arm in ("left", "right"):
                arm_qpos = qpos[ARM_SLICES[arm]]
                pose_world = self.calibration.pose_to_world(
                    arm,
                    self._kinematics_for(arm).fk(arm_qpos[:6], float(arm_qpos[6])),
                )
                clearance = float(pose_world[2, 3] - self.table_z)
                min_clearance = min(min_clearance, clearance)
        return {
            "ok": min_clearance >= self.table_clearance_m,
            "checked": True,
            "reason": None
            if min_clearance >= self.table_clearance_m
            else "table_guard",
            "min_tcp_clearance_m": min_clearance,
            "required_clearance_m": self.table_clearance_m,
            "limitation": "checks TCP height only; no self, object, or fixture collision guarantee",
        }

    def _check_table_guard(
        self,
        arm: Literal["left", "right"],
        position: np.ndarray,
        gripper: float,
        kin: Any,
    ) -> dict[str, Any]:
        if self.table_z is None:
            return {
                "ok": False,
                "checked": False,
                "reason": "table_not_configured",
                "limitation": "table_z is required before executable Cartesian plans",
            }
        min_clearance = float("inf")
        for q in position:
            pose_world = self.calibration.pose_to_world(arm, kin.fk(q, gripper))
            clearance = float(pose_world[2, 3] - self.table_z)
            min_clearance = min(min_clearance, clearance)
        return {
            "ok": min_clearance >= self.table_clearance_m,
            "checked": True,
            "reason": None
            if min_clearance >= self.table_clearance_m
            else "table_guard",
            "min_tcp_clearance_m": min_clearance,
            "required_clearance_m": self.table_clearance_m,
            "limitation": "checks TCP height only; no self, object, or fixture collision guarantee",
        }
