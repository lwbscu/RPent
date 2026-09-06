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

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rpent.utils.config import get_rlinf_repo_path

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


def _load_json_path(value: str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    return json.loads(Path(value).expanduser().read_text())


@dataclass(frozen=True)
class YamCalibration:
    """Transforms into the RPent world frame, defined as the left robot base."""

    camera_to_world: dict[str, np.ndarray]
    world_from_right_base: np.ndarray | None
    wrist_grasp_from_camera: dict[str, np.ndarray]
    world_frame: str = "left_base"

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
                "right-arm planning requires extrinsics_path with "
                "top_camera.T_base_to_cam via both arms"
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
    """Load the RLinf solve_handeye extrinsics.json producer format."""
    config = config or {}
    calibration = _load_json_path(config.get("extrinsics_path"))
    camera_to_world: dict[str, np.ndarray] = {}
    wrist_grasp_from_camera: dict[str, np.ndarray] = {}
    right_base_from_top: np.ndarray | None = None

    top_transforms = calibration.get("top_camera", {}).get("T_base_to_cam", {})
    if "via_left_arm" in top_transforms:
        camera_to_world["top"] = matrix4(
            top_transforms["via_left_arm"],
            name="top_camera.T_base_to_cam.via_left_arm",
        )
    if "via_right_arm" in top_transforms:
        right_base_from_top = matrix4(
            top_transforms["via_right_arm"],
            name="top_camera.T_base_to_cam.via_right_arm",
        )

    for key, arm in (("left_wrist", "left"), ("right_wrist", "right")):
        values = calibration.get(key)
        if isinstance(values, dict) and values.get("T_grasp_to_cam") is not None:
            # RLinf solve_handeye.py verifies FK @ t_tcp_cam @ board_in_cam
            # and writes t_tcp_cam under this historical key. Do not invert.
            wrist_grasp_from_camera[arm] = matrix4(
                values["T_grasp_to_cam"], name=f"{key}.T_grasp_to_cam"
            )

    world_from_right = None
    if "top" in camera_to_world and right_base_from_top is not None:
        world_from_right = camera_to_world["top"] @ np.linalg.inv(right_base_from_top)

    return YamCalibration(
        camera_to_world=camera_to_world,
        world_from_right_base=world_from_right,
        wrist_grasp_from_camera=wrist_grasp_from_camera,
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
        rlinf_root = get_rlinf_repo_path()
        if rlinf_root is not None and str(rlinf_root) not in sys.path:
            sys.path.insert(0, str(rlinf_root))
        try:
            from rlinf.envs.realworld.yam.kinematics import YamKinematicsAdapter
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "missing dependency while importing RLinf YAM kinematics. "
                "Set RPENT_RLINF_ROOT or RLINF_REPO_PATH to the RLinf checkout "
                "and install the pinned YAM/i2rt dependencies on the robot host."
            ) from error
        adapter = YamKinematicsAdapter(
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
            meta["cam2world_source"] = "static_top_extrinsics"
            return meta
        if name not in {"left", "right"}:
            raise ValueError(f"unknown YAM camera {camera_name!r}")
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        arm_qpos = qpos[ARM_SLICES[name]]
        grasp_from_camera = self.calibration.wrist_grasp_from_camera.get(name)
        if grasp_from_camera is None:
            raise ValueError(
                f"{name} wrist camera requires dynamic hand-eye calibration: "
                f"set extrinsics_path to an RLinf solve_handeye output with "
                f"{name}_wrist.T_grasp_to_cam"
            )
        base_from_grasp = self._kinematics_for(name).fk(
            arm_qpos[:6], float(arm_qpos[6])
        )
        world_from_camera = self.calibration.pose_to_world(
            name, base_from_grasp
        ) @ matrix4(grasp_from_camera, name=f"{name}_wrist.T_grasp_to_cam")
        meta["cam2world_cv"] = world_from_camera
        meta["cam2world_source"] = "dynamic_fk_current_qpos_approx_cached_frame"
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
        if table_check["checked"] and not table_check["ok"]:
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
                "ok": True,
                "checked": False,
                "reason": None,
                "limitation": "table_z is not configured; table guard disabled",
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
                "ok": True,
                "checked": False,
                "reason": None,
                "limitation": "table_z is not configured; table guard disabled",
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
