"""Public-target geometry for a fixed opposite-wrist visual observer.

The transform constants below come from the installed official R1Pro USD.  They
describe robot link geometry only; no task-object pose or simulator state is
used.  Kit cameras look along local ``-Z`` with local ``+Y`` as image up.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# R1Pro USD PhysicsFixedJoint values use quaternion order (w, x, y, z).
_REALSENSE_LOCAL0_POSITION = (0.05051, 0.0028934, 0.0051317)
_REALSENSE_LOCAL0_QUAT_WXYZ = (
    0.8433849,
    -0.0015274044,
    -0.53730685,
    0.00096990325,
)
_REALSENSE_LOCAL1_QUAT_WXYZ = (0.70710677, 0.0, 0.70710677, 0.0)
_CAMERA_IN_REALSENSE_QUAT_WXYZ = (
    0.0,
    0.707099974155426,
    -0.707099974155426,
    0.0,
)
_EEF_IN_GRIPPER_POSITION = (0.0, 0.0, -0.06)
_EEF_IN_GRIPPER_QUAT_WXYZ = (0.0, 0.0, 1.0, 0.0)


def _finite_vector(value: Sequence[float], *, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _quat_wxyz_to_matrix(value: Sequence[float]) -> np.ndarray:
    quat = _finite_vector(value, size=4, label="quaternion")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be finite")
    # Stable branch formulation, returning the public API's (x, y, z, w).
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = (
                math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = (
                math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = (
                math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def _pose_matrix(position: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz)
    pose[:3, 3] = _finite_vector(position, size=3, label="position")
    return pose


def r1pro_eef_to_wrist_camera_transform() -> np.ndarray:
    """Return the fixed R1Pro EEF-to-Kit-camera transform for either wrist."""

    gripper_to_camera_link = _pose_matrix(
        _REALSENSE_LOCAL0_POSITION,
        _REALSENSE_LOCAL0_QUAT_WXYZ,
    ) @ np.linalg.inv(_pose_matrix((0.0, 0.0, 0.0), _REALSENSE_LOCAL1_QUAT_WXYZ))
    camera_link_to_camera = _pose_matrix(
        (0.0, 0.0, 0.0),
        _CAMERA_IN_REALSENSE_QUAT_WXYZ,
    )
    gripper_to_eef = _pose_matrix(
        _EEF_IN_GRIPPER_POSITION,
        _EEF_IN_GRIPPER_QUAT_WXYZ,
    )
    return (
        np.linalg.inv(gripper_to_eef) @ gripper_to_camera_link @ camera_link_to_camera
    )


def opposite_wrist_observer_pose(
    surface_target_xyz: Sequence[float],
    *,
    look_offset_world_m: Sequence[float] = (0.0, 0.0, -0.10),
    camera_offset_world_m: Sequence[float] = (0.0, -0.45, 0.02),
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Compute a table-height wrist observer pose from fresh public RGB-D.

    ``surface_target_xyz`` is the output of public ``pixel_to_world``.  The
    fixed offsets intentionally place the camera far enough away to show the
    complete radio, gripper, original footprint, and table gap in one view.
    """

    surface = _finite_vector(surface_target_xyz, size=3, label="surface target")
    look_at = surface + _finite_vector(look_offset_world_m, size=3, label="look offset")
    camera_position = look_at + _finite_vector(
        camera_offset_world_m,
        size=3,
        label="camera offset",
    )
    forward = look_at - camera_position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-6:
        raise ValueError("camera position and look-at point must differ")
    forward /= forward_norm
    camera_z = -forward  # Kit camera optical axis is local -Z.
    up = _finite_vector(world_up, size=3, label="world up")
    camera_x = np.cross(up, camera_z)
    x_norm = float(np.linalg.norm(camera_x))
    if x_norm <= 1e-6:
        raise ValueError("world up must not be parallel to the camera optical axis")
    camera_x /= x_norm
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.column_stack((camera_x, camera_y, camera_z))
    world_to_camera[:3, 3] = camera_position
    eef_to_camera = r1pro_eef_to_wrist_camera_transform()
    world_to_eef = world_to_camera @ np.linalg.inv(eef_to_camera)

    reconstructed_camera = world_to_eef @ eef_to_camera
    optical_axis = -reconstructed_camera[:3, 2]
    optical_alignment = float(np.dot(optical_axis, forward))
    if optical_alignment < 1.0 - 1e-8:
        raise RuntimeError("observer camera optical-axis reconstruction failed")

    return {
        "source": "fresh_public_pixel_to_world_plus_official_r1pro_fixed_extrinsic",
        "surface_target_xyz": surface.tolist(),
        "look_offset_world_m": _finite_vector(
            look_offset_world_m,
            size=3,
            label="look offset",
        ).tolist(),
        "look_at_world_xyz": look_at.tolist(),
        "camera_offset_world_m": _finite_vector(
            camera_offset_world_m,
            size=3,
            label="camera offset",
        ).tolist(),
        "camera_position_world_xyz": camera_position.tolist(),
        "camera_quat_xyzw": _matrix_to_quat_xyzw(world_to_camera[:3, :3]).tolist(),
        "eef_position_world_xyz": world_to_eef[:3, 3].tolist(),
        "eef_quat_xyzw": _matrix_to_quat_xyzw(world_to_eef[:3, :3]).tolist(),
        "camera_forward_world": forward.tolist(),
        "optical_alignment": optical_alignment,
        "eef_to_camera": eef_to_camera.tolist(),
    }
