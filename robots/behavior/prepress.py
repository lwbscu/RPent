"""Fail-closed visual and geometric gates for BEHAVIOR radio pre-press."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

PREPRESS_LINE_DISTANCE_MAX_M = 0.010
PREPRESS_OPPOSITION_ANGLE_MAX_DEG = 15.0
PREPRESS_AXIAL_STANDOFF_MIN_M = 0.03
PREPRESS_AXIAL_STANDOFF_MAX_M = 0.06
# `press_staging` is also the button-goal-driven observation move used before
# the final close pre-press alignment.  A wrist-camera staging pose may need to
# remain farther away for arm reachability and collision clearance; the strict
# state_checkpoint_2 geometry gate above remains 0.03--0.06 m.
PRESS_STAGING_AXIAL_STANDOFF_MAX_M = 0.25

BUTTON_FACE_CLASS = "BUTTON_FACE"
CLEAR_SLOTTED_BACK_FACE_CLASS = "CLEAR_SLOTTED_BACK_FACE"
SIDE_PORT_FACE_CLASS = "SIDE_PORT"
AMBIGUOUS_FACE_CLASS = "AMBIGUOUS"
NEGATIVE_CASE_TO_FACE_CLASS = {
    "clear_slotted_back_face": CLEAR_SLOTTED_BACK_FACE_CLASS,
    "side_port": SIDE_PORT_FACE_CLASS,
    "ambiguous": AMBIGUOUS_FACE_CLASS,
}

# Uncertain views may only make bounded observation-improving motions.  A clear
# slotted broad back face authorizes one larger held-hand back-to-front move;
# a positive button face authorizes only final visual/geometric refinement.
UNCERTAIN_SEARCH_MAX_TRANSLATION_M = 0.05
UNCERTAIN_SEARCH_MAX_ROTATION_RAD = math.radians(15.0)
DIRECT_BACK_FACE_MAX_TRANSLATION_M = 0.60
DIRECT_BACK_FACE_MAX_ROTATION_RAD = math.pi
DIRECT_BACK_FACE_ALIGNMENT_MAX_DEG = 30.0
DIRECT_BACK_FACE_UPRIGHT_MAX_DEG = 20.0
FINAL_MICRO_ADJUST_MAX_TRANSLATION_M = 0.08
FINAL_MICRO_ADJUST_MAX_ROTATION_RAD = math.radians(30.0)
# The press hand is empty during stage 2.  Do not reuse the held-radio visual
# search limits for its non-contact staging move; reachability and collision
# safety are decided by the mandatory plan-only CuRobo trajectory instead.
PRESS_STAGING_MAX_TRANSLATION_M = 1.0
PRESS_STAGING_MAX_ROTATION_RAD = math.pi
# Before the press wrist can see the button, it may rotate in place to acquire
# that view.  This is deliberately distinct from press staging: it grants no
# translational freedom and therefore cannot be used to approach the radio.
PRESS_OBSERVATION_ROTATION_MAX_TRANSLATION_M = 0.0
PRESS_OBSERVATION_ROTATION_MAX_ROTATION_RAD = math.pi
SAFETY_RAISE_MIN_Z_M = 0.04
SAFETY_RAISE_MAX_TRANSLATION_M = 0.15

# Task-specific radio_89 geometry, expressed in the radio root frame.  The
# center was cross-checked from four independent positive button projections;
# the face normal is the corresponding broad-face direction with the noisy
# bump-normal z component removed.  These are a coarse reveal/alignment prior,
# never a substitute for a fresh positive button gate and back-projection.
RADIO_LOCAL_BUTTON_CENTER_M = (0.05292, 0.03502, -0.01339)
RADIO_LOCAL_BUTTON_FACE_NORMAL = (0.7740, -0.6332, 0.0)
RADIO_LOCAL_UP_AXIS = (0.0, 0.0, 1.0)

POSITIVE_SIGNATURE_FIELDS = (
    "red_front_face",
    "black_round_or_oval_disk",
    "white_outer_ring",
    "red_center_bump",
)


def _unit_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinity")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} has zero length")
    return vector / norm


def quat_rotate_xyzw(quaternion: Any, vector: Any) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(quat).all() or float(np.linalg.norm(quat)) <= 1e-9:
        raise ValueError("quaternion is invalid")
    quat /= np.linalg.norm(quat)
    xyz = np.asarray(vector, dtype=np.float64).reshape(3)
    qxyz = quat[:3]
    return xyz + 2.0 * (
        quat[3] * np.cross(qxyz, xyz) + np.cross(qxyz, np.cross(qxyz, xyz))
    )


def quat_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    """Compose two xyzw quaternions, applying right then left."""

    lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    return result / max(float(np.linalg.norm(result)), 1e-12)


def quat_between_vectors_xyzw(source: Any, target: Any) -> np.ndarray:
    """Return the shortest world rotation from source to target."""

    source_unit = _unit_vector(source, name="source_vector")
    target_unit = _unit_vector(target, name="target_vector")
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot < -1.0 + 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(source_unit, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source_unit, basis)
        axis /= np.linalg.norm(axis)
        return np.r_[axis, 0.0]
    quaternion = np.r_[np.cross(source_unit, target_unit), 1.0 + dot]
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def pose_matrix_xyzw(position: Any, quaternion: Any) -> np.ndarray:
    """Return a row-major rigid transform mapping local points to world."""

    position_array = np.asarray(position, dtype=np.float64).reshape(3)
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(position_array).all() or not np.isfinite(quat).all():
        raise ValueError("pose contains NaN or infinity")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        raise ValueError("pose quaternion has zero length")
    x, y, z, w = quat / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position_array
    return transform


def quaternion_xyzw_from_rotation_matrix(rotation: Any) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix into a normalized xyzw quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("rotation matrix contains NaN or infinity")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-6
    ):
        raise ValueError("rotation matrix is not a proper orthonormal rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat = -quat
    return quat


def upright_radio_orientation_xyzw(
    target_front_normal_world: Any,
    *,
    local_front_normal: Any = RADIO_LOCAL_BUTTON_FACE_NORMAL,
    local_up_axis: Any = RADIO_LOCAL_UP_AXIS,
) -> np.ndarray:
    """Build a unique radio orientation from a target front normal and upright axis.

    The second axis removes the antiparallel/180-degree ambiguity that a
    shortest-arc single-vector rotation cannot resolve safely.
    """

    local_front = _unit_vector(local_front_normal, name="local_front_normal")
    local_up_seed = _unit_vector(local_up_axis, name="local_up_axis")
    local_up = local_up_seed - np.dot(local_up_seed, local_front) * local_front
    local_up = _unit_vector(local_up, name="local_up_in_face_plane")
    local_right = _unit_vector(np.cross(local_up, local_front), name="local_face_right")

    world_front = _unit_vector(
        target_front_normal_world, name="target_front_normal_world"
    )
    world_up_seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    world_up = world_up_seed - np.dot(world_up_seed, world_front) * world_front
    world_up = _unit_vector(world_up, name="world_up_in_face_plane")
    world_right = _unit_vector(np.cross(world_up, world_front), name="world_face_right")
    local_frame = np.column_stack([local_right, local_up, local_front])
    world_frame = np.column_stack([world_right, world_up, world_front])
    return quaternion_xyzw_from_rotation_matrix(world_frame @ local_frame.T)


def direct_back_to_front_alignment(
    *,
    target_held_pose: Any,
    held_to_radio_transform: Any,
    press_eef_position_world: Any,
) -> dict[str, Any]:
    """Validate that a coarse held target reveals an upright front toward press."""

    target_held = np.asarray(target_held_pose, dtype=np.float64).reshape(4, 4)
    held_to_radio = np.asarray(held_to_radio_transform, dtype=np.float64).reshape(4, 4)
    press_position = np.asarray(press_eef_position_world, dtype=np.float64).reshape(3)
    if not (
        np.isfinite(target_held).all()
        and np.isfinite(held_to_radio).all()
        and np.isfinite(press_position).all()
    ):
        raise ValueError("direct back-to-front transform contains NaN or infinity")
    target_radio = target_held @ held_to_radio
    rotation = target_radio[:3, :3]
    button_center = (
        target_radio[:3, :3] @ np.asarray(RADIO_LOCAL_BUTTON_CENTER_M, dtype=np.float64)
        + target_radio[:3, 3]
    )
    front = _unit_vector(
        rotation @ np.asarray(RADIO_LOCAL_BUTTON_FACE_NORMAL),
        name="predicted_front_normal",
    )
    radio_up = _unit_vector(
        rotation @ np.asarray(RADIO_LOCAL_UP_AXIS),
        name="predicted_radio_up",
    )
    toward_press = _unit_vector(
        press_position - button_center, name="button_to_press_hand"
    )
    desired_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    desired_up -= np.dot(desired_up, front) * front
    desired_up = _unit_vector(desired_up, name="desired_upright_axis")
    face_angle = math.degrees(
        math.acos(float(np.clip(np.dot(front, toward_press), -1.0, 1.0)))
    )
    upright_angle = math.degrees(
        math.acos(float(np.clip(np.dot(radio_up, desired_up), -1.0, 1.0)))
    )
    criteria = {
        "front_points_toward_press": face_angle <= DIRECT_BACK_FACE_ALIGNMENT_MAX_DEG,
        "radio_upright": upright_angle <= DIRECT_BACK_FACE_UPRIGHT_MAX_DEG,
    }
    return {
        "valid": bool(all(criteria.values())),
        "criteria": criteria,
        "front_to_press_angle_deg": face_angle,
        "upright_error_deg": upright_angle,
        "target_radio_pose": target_radio.tolist(),
        "predicted_button_center_world": button_center.tolist(),
        "predicted_front_normal_world": front.tolist(),
        "predicted_radio_up_world": radio_up.tolist(),
        "thresholds": {
            "max_front_to_press_angle_deg": DIRECT_BACK_FACE_ALIGNMENT_MAX_DEG,
            "max_upright_error_deg": DIRECT_BACK_FACE_UPRIGHT_MAX_DEG,
        },
    }


def _rigid_transform(value: Any, *, name: str) -> np.ndarray:
    """Return one finite, proper, row-major 4x4 rigid transform."""

    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} is not a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} rotation is not proper orthonormal")
    return transform


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        second, dtype=np.float64
    ).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _vector_angle_deg(first: Any, second: Any) -> float:
    left = _unit_vector(first, name="first_vector")
    right = _unit_vector(second, name="second_vector")
    return math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def _axis_angle_quaternion_xyzw(value: Any) -> tuple[np.ndarray, float]:
    command = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.isfinite(command).all():
        raise ValueError("orientation perturbation contains NaN or infinity")
    angle = float(command[3])
    if abs(angle) <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0]), 0.0
    axis = _unit_vector(command[:3], name="orientation_perturbation_axis")
    half = 0.5 * angle
    return np.r_[axis * math.sin(half), math.cos(half)], abs(angle)


def _side_view_normal(
    face_to_press: np.ndarray,
    head_optical_axis: np.ndarray,
    fallback_normal: np.ndarray,
) -> np.ndarray:
    """Return the closest face-to-press direction orthogonal to the head axis."""

    projected = face_to_press - np.dot(face_to_press, head_optical_axis) * (
        head_optical_axis
    )
    if float(np.linalg.norm(projected)) <= 1e-9:
        projected = (
            fallback_normal
            - np.dot(fallback_normal, head_optical_axis) * head_optical_axis
        )
    if float(np.linalg.norm(projected)) <= 1e-9:
        for basis in np.eye(3):
            projected = basis - np.dot(basis, head_optical_axis) * head_optical_axis
            if float(np.linalg.norm(projected)) > 1e-9:
                break
    side = _unit_vector(projected, name="side_view_normal")
    return -side if float(np.dot(side, face_to_press)) < 0.0 else side


def generate_button_goal_pose_candidates(
    *,
    world_held_transform: Any,
    world_radio_transform: Any,
    held_to_radio_transform: Any,
    button_center_world: Any,
    button_normal_world: Any,
    world_press_transform: Any,
    goal: dict[str, Any],
    head_optical_axis_world: Any | None = None,
) -> dict[str, Any]:
    """Generate deterministic radio and held-EEF candidates from button goals.

    This is a pure geometric proposal function: it neither chooses a physical
    hand nor calls a planner.  ``goal`` requires ``chest_direction_world`` and
    may provide a net ``chest_translation_m``, bounded world-position and
    world-axis-angle perturbations, and normal blend factors.  Blend zero faces
    the measured button toward the current press EEF; blend one is the closest
    such normal that is 90 degrees from the optional head optical axis.

    Every desired held pose is recovered with
    ``T_world_held_target = T_world_radio_desired @ inv(T_held_radio)``.
    Scores are geometric ranking signals only, never collision or reachability
    certificates.
    """

    if not isinstance(goal, dict):
        raise ValueError("goal must be a dictionary")
    allowed_goal_keys = {
        "chest_direction_world",
        "chest_translation_m",
        "position_perturbations_world_m",
        "orientation_perturbations_world_axis_angle",
        "normal_blend_factors",
        "max_position_perturbation_m",
        "max_orientation_perturbation_rad",
        "max_face_to_press_angle_deg",
        "max_press_approach_opposition_angle_deg",
        "target_head_side_angle_deg",
        "max_head_side_error_deg",
        "chest_translation_tolerance_m",
        "transform_consistency_position_tolerance_m",
        "transform_consistency_orientation_rad",
        "local_up_axis",
        "press_approach_axis_local",
        "score_weights",
        "max_candidates",
        "orientation_goal",
    }
    unknown = set(goal) - allowed_goal_keys
    if unknown:
        raise ValueError(f"unsupported button-goal keys: {sorted(unknown)}")
    if "chest_direction_world" not in goal:
        raise ValueError("goal.chest_direction_world is required")

    world_held = _rigid_transform(world_held_transform, name="world_held_transform")
    world_radio = _rigid_transform(world_radio_transform, name="world_radio_transform")
    held_to_radio = _rigid_transform(
        held_to_radio_transform, name="held_to_radio_transform"
    )
    world_press = _rigid_transform(world_press_transform, name="world_press_transform")
    reconstructed_radio = world_held @ held_to_radio
    consistency_position_error = float(
        np.linalg.norm(reconstructed_radio[:3, 3] - world_radio[:3, 3])
    )
    consistency_orientation_error = _rotation_distance_rad(
        reconstructed_radio[:3, :3], world_radio[:3, :3]
    )

    position_consistency_limit = float(
        goal.get("transform_consistency_position_tolerance_m", 0.01)
    )
    orientation_consistency_limit = float(
        goal.get("transform_consistency_orientation_rad", math.radians(5.0))
    )
    if (
        not math.isfinite(position_consistency_limit)
        or position_consistency_limit < 0.0
        or not math.isfinite(orientation_consistency_limit)
        or orientation_consistency_limit < 0.0
    ):
        raise ValueError(
            "transform consistency tolerances must be finite and non-negative"
        )
    if (
        consistency_position_error > position_consistency_limit
        or consistency_orientation_error > orientation_consistency_limit
    ):
        raise ValueError("world held/radio transforms disagree with held_to_radio")

    button_world = np.asarray(button_center_world, dtype=np.float64).reshape(3)
    if not np.isfinite(button_world).all():
        raise ValueError("button_center_world contains NaN or infinity")
    button_normal = _unit_vector(button_normal_world, name="button_normal_world")
    radio_rotation = world_radio[:3, :3]
    button_center_local = radio_rotation.T @ (button_world - world_radio[:3, 3])
    button_normal_local = _unit_vector(
        radio_rotation.T @ button_normal,
        name="button_normal_radio_local",
    )
    local_up = _unit_vector(
        goal.get("local_up_axis", RADIO_LOCAL_UP_AXIS), name="local_up_axis"
    )
    if abs(float(np.dot(local_up, button_normal_local))) >= 1.0 - 1e-6:
        raise ValueError("local_up_axis is parallel to the button normal")

    raw_chest_direction = np.asarray(
        goal["chest_direction_world"], dtype=np.float64
    ).reshape(-1)
    if raw_chest_direction.size == 2:
        raw_chest_direction = np.r_[raw_chest_direction, 0.0]
    if raw_chest_direction.size != 3:
        raise ValueError("chest_direction_world must contain two or three values")
    chest_direction = _unit_vector(raw_chest_direction, name="chest_direction_world")
    chest_translation = float(goal.get("chest_translation_m", 0.10))
    max_position_perturbation = float(goal.get("max_position_perturbation_m", 0.05))
    max_orientation_perturbation = float(
        goal.get("max_orientation_perturbation_rad", math.radians(30.0))
    )
    face_limit_deg = float(goal.get("max_face_to_press_angle_deg", 30.0))
    press_approach_limit_deg = float(
        goal.get("max_press_approach_opposition_angle_deg", 30.0)
    )
    side_target_deg = float(goal.get("target_head_side_angle_deg", 90.0))
    side_limit_deg = float(goal.get("max_head_side_error_deg", 20.0))
    chest_tolerance = float(goal.get("chest_translation_tolerance_m", 0.02))
    scalar_values = {
        "chest_translation_m": chest_translation,
        "max_position_perturbation_m": max_position_perturbation,
        "max_orientation_perturbation_rad": max_orientation_perturbation,
        "max_face_to_press_angle_deg": face_limit_deg,
        "max_press_approach_opposition_angle_deg": press_approach_limit_deg,
        "target_head_side_angle_deg": side_target_deg,
        "max_head_side_error_deg": side_limit_deg,
        "chest_translation_tolerance_m": chest_tolerance,
    }
    if not all(math.isfinite(value) for value in scalar_values.values()):
        raise ValueError("button-goal scalars must be finite")
    if (
        chest_translation < 0.0
        or max_position_perturbation < 0.0
        or max_orientation_perturbation < 0.0
        or face_limit_deg <= 0.0
        or press_approach_limit_deg <= 0.0
        or not 0.0 <= side_target_deg <= 180.0
        or side_limit_deg <= 0.0
        or chest_tolerance <= 0.0
    ):
        raise ValueError("button-goal scalar limits are invalid")

    position_perturbations = goal.get(
        "position_perturbations_world_m", [[0.0, 0.0, 0.0]]
    )
    orientation_perturbations = goal.get(
        "orientation_perturbations_world_axis_angle", [[0.0, 0.0, 1.0, 0.0]]
    )
    if not isinstance(position_perturbations, (list, tuple)) or not (
        position_perturbations
    ):
        raise ValueError("position perturbations must be a non-empty list")
    if not isinstance(orientation_perturbations, (list, tuple)) or not (
        orientation_perturbations
    ):
        raise ValueError("orientation perturbations must be a non-empty list")
    positions: list[tuple[np.ndarray, float]] = []
    for value in position_perturbations:
        perturbation = np.asarray(value, dtype=np.float64).reshape(3)
        if not np.isfinite(perturbation).all():
            raise ValueError("position perturbation contains NaN or infinity")
        magnitude = float(np.linalg.norm(perturbation))
        if magnitude > max_position_perturbation + 1e-9:
            raise ValueError("position perturbation exceeds its declared bound")
        positions.append((perturbation, magnitude))
    orientations: list[tuple[np.ndarray, float]] = []
    for value in orientation_perturbations:
        quaternion, magnitude = _axis_angle_quaternion_xyzw(value)
        if magnitude > max_orientation_perturbation + 1e-9:
            raise ValueError("orientation perturbation exceeds its declared bound")
        orientations.append((quaternion, magnitude))

    head_axis = (
        None
        if head_optical_axis_world is None
        else _unit_vector(head_optical_axis_world, name="head_optical_axis_world")
    )
    default_blends = [0.0, 0.5, 1.0] if head_axis is not None else [0.0]
    raw_blends = goal.get("normal_blend_factors", default_blends)
    if not isinstance(raw_blends, (list, tuple)) or not raw_blends:
        raise ValueError("normal_blend_factors must be a non-empty list")
    blends = [float(value) for value in raw_blends]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in blends):
        raise ValueError("normal blend factors must lie within [0, 1]")
    if head_axis is None and any(value != 0.0 for value in blends):
        raise ValueError("non-zero normal blend requires head_optical_axis_world")
    orientation_goal = goal.get("orientation_goal", "side_to_press")
    if orientation_goal not in {"side_to_press", "preserve_current"}:
        raise ValueError("orientation_goal must be side_to_press or preserve_current")

    default_weights = {
        "face": 3.0,
        "press_approach": 3.0,
        "side": 2.0 if head_axis is not None else 0.0,
        "chest": 2.0,
        "position_perturbation": 0.25,
        "orientation_perturbation": 0.25,
        "held_translation": 0.25,
        "held_rotation": 0.25,
    }
    supplied_weights = goal.get("score_weights", {})
    if not isinstance(supplied_weights, dict) or set(supplied_weights) - set(
        default_weights
    ):
        raise ValueError("score_weights contains unsupported keys")
    weights = {**default_weights, **supplied_weights}
    weights = {name: float(value) for name, value in weights.items()}
    if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise ValueError("score weights must be finite and non-negative")

    raw_max_candidates = goal.get("max_candidates", 32)
    if isinstance(raw_max_candidates, bool) or not isinstance(
        raw_max_candidates, (int, np.integer)
    ):
        raise ValueError("max_candidates must be an integer in [1, 128]")
    max_candidates = int(raw_max_candidates)
    if not 1 <= max_candidates <= 128:
        raise ValueError("max_candidates must be an integer in [1, 128]")
    combination_count = len(positions) * len(orientations) * len(blends)
    if combination_count > max_candidates:
        raise ValueError("button-goal candidate product exceeds max_candidates")

    current_held_quat = quaternion_xyzw_from_rotation_matrix(world_held[:3, :3])
    press_position = world_press[:3, 3]
    press_approach_axis = _unit_vector(
        goal.get("press_approach_axis_local", [0.0, 0.0, 1.0]),
        name="press_approach_axis_local",
    )
    press_direction = _unit_vector(
        world_press[:3, :3] @ press_approach_axis,
        name="press_direction_world",
    )
    nominal_button_position = button_world + chest_translation * chest_direction
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for position_index, (position_delta, position_magnitude) in enumerate(positions):
        target_button_position = nominal_button_position + position_delta
        for blend_index, blend in enumerate(blends):
            face_to_press = _unit_vector(
                press_position - target_button_position,
                name="target_button_to_press",
            )
            if orientation_goal == "preserve_current":
                desired_normal = button_normal
                nominal_quat = quaternion_xyzw_from_rotation_matrix(world_radio[:3, :3])
            elif head_axis is None:
                desired_normal = face_to_press
                nominal_quat = upright_radio_orientation_xyzw(
                    desired_normal,
                    local_front_normal=button_normal_local,
                    local_up_axis=local_up,
                )
            else:
                side_normal = _side_view_normal(face_to_press, head_axis, button_normal)
                desired_normal = _unit_vector(
                    (1.0 - blend) * face_to_press + blend * side_normal,
                    name="blended_button_goal_normal",
                )
                nominal_quat = upright_radio_orientation_xyzw(
                    desired_normal,
                    local_front_normal=button_normal_local,
                    local_up_axis=local_up,
                )

            for orientation_index, (
                orientation_delta,
                orientation_magnitude,
            ) in enumerate(orientations):
                target_radio_quat = quat_multiply_xyzw(orientation_delta, nominal_quat)
                target_rotation = pose_matrix_xyzw([0.0, 0.0, 0.0], target_radio_quat)[
                    :3, :3
                ]
                target_radio_position = target_button_position - (
                    target_rotation @ button_center_local
                )
                desired_radio = pose_matrix_xyzw(
                    target_radio_position, target_radio_quat
                )
                target_held = desired_radio @ np.linalg.inv(held_to_radio)
                target_held_quat = quaternion_xyzw_from_rotation_matrix(
                    target_held[:3, :3]
                )
                candidate_button = (
                    target_radio_position + target_rotation @ button_center_local
                )
                candidate_normal = _unit_vector(
                    target_rotation @ button_normal_local,
                    name="candidate_button_normal_world",
                )
                candidate_up = _unit_vector(
                    target_rotation @ local_up, name="candidate_radio_up_world"
                )
                button_to_press = _unit_vector(
                    press_position - candidate_button,
                    name="candidate_button_to_press",
                )
                face_angle = _vector_angle_deg(candidate_normal, button_to_press)
                press_opposition_angle = _vector_angle_deg(
                    candidate_normal, -press_direction
                )
                head_angle = (
                    None
                    if head_axis is None
                    else _vector_angle_deg(candidate_normal, head_axis)
                )
                head_side_error = (
                    None if head_angle is None else abs(head_angle - side_target_deg)
                )
                button_delta = candidate_button - button_world
                achieved_chest_translation = float(
                    np.dot(button_delta, chest_direction)
                )
                chest_error = abs(achieved_chest_translation - chest_translation)
                chest_lateral_drift = float(
                    np.linalg.norm(
                        button_delta - achieved_chest_translation * chest_direction
                    )
                )
                held_translation = float(
                    np.linalg.norm(target_held[:3, 3] - world_held[:3, 3])
                )
                held_rotation = quaternion_angle_rad(
                    current_held_quat, target_held_quat
                )
                normalized_costs = {
                    "face": (
                        0.0
                        if orientation_goal == "preserve_current"
                        else face_angle / face_limit_deg
                    ),
                    "press_approach": (
                        0.0
                        if orientation_goal == "preserve_current"
                        else press_opposition_angle / press_approach_limit_deg
                    ),
                    "side": (
                        0.0
                        if head_side_error is None
                        or orientation_goal == "preserve_current"
                        else head_side_error / side_limit_deg
                    ),
                    "chest": chest_error / chest_tolerance,
                    "position_perturbation": position_magnitude
                    / max(max_position_perturbation, 1e-9),
                    "orientation_perturbation": orientation_magnitude
                    / max(max_orientation_perturbation, 1e-9),
                    "held_translation": held_translation
                    / DIRECT_BACK_FACE_MAX_TRANSLATION_M,
                    "held_rotation": held_rotation / DIRECT_BACK_FACE_MAX_ROTATION_RAD,
                }
                weighted_cost = float(
                    sum(
                        weights[name] * normalized_costs[name]
                        for name in default_weights
                    )
                )
                geometry_score = 1.0 / (1.0 + weighted_cost)
                criteria = {
                    "face_toward_press": (
                        orientation_goal == "preserve_current"
                        or face_angle <= face_limit_deg
                    ),
                    "opposes_press_approach": (
                        orientation_goal == "preserve_current"
                        or press_opposition_angle <= press_approach_limit_deg
                    ),
                    "head_side_view": (
                        orientation_goal == "preserve_current"
                        or head_side_error is None
                        or head_side_error <= side_limit_deg
                    ),
                    "chest_translation": chest_error <= chest_tolerance,
                    "position_perturbation_bounded": (
                        position_magnitude <= max_position_perturbation + 1e-9
                    ),
                    "orientation_perturbation_bounded": (
                        orientation_magnitude <= max_orientation_perturbation + 1e-9
                    ),
                    "held_motion_within_direct_limits": (
                        held_translation <= DIRECT_BACK_FACE_MAX_TRANSLATION_M
                        and held_rotation <= DIRECT_BACK_FACE_MAX_ROTATION_RAD
                    ),
                }
                dedup_key = tuple(
                    np.round(
                        np.r_[target_radio_position, target_radio_quat], 10
                    ).tolist()
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                candidates.append(
                    {
                        "candidate_id": (
                            f"button_goal_p{position_index:02d}_"
                            f"b{blend_index:02d}_o{orientation_index:02d}"
                        ),
                        "eligible": bool(all(criteria.values())),
                        "normal_blend_factor": blend,
                        "desired_radio_pose": {
                            "position": target_radio_position.tolist(),
                            "quat_xyzw": target_radio_quat.tolist(),
                            "matrix": desired_radio.tolist(),
                        },
                        "target_held_eef_pose": {
                            "position": target_held[:3, 3].tolist(),
                            "quat_xyzw": target_held_quat.tolist(),
                            "matrix": target_held.tolist(),
                            "construction": (
                                "T_world_radio_desired_times_"
                                "inverse_T_held_radio_current"
                            ),
                        },
                        "predicted_button_center_world": candidate_button.tolist(),
                        "predicted_button_normal_world": candidate_normal.tolist(),
                        "predicted_radio_up_world": candidate_up.tolist(),
                        "criteria": criteria,
                        "geometry": {
                            "face_to_press_angle_deg": face_angle,
                            "press_approach_opposition_angle_deg": (
                                press_opposition_angle
                            ),
                            "head_normal_angle_deg": head_angle,
                            "head_side_view_error_deg": head_side_error,
                            "achieved_chest_translation_m": (
                                achieved_chest_translation
                            ),
                            "chest_translation_error_m": chest_error,
                            "chest_lateral_drift_m": chest_lateral_drift,
                            "position_perturbation_m": position_magnitude,
                            "orientation_perturbation_rad": (orientation_magnitude),
                            "held_translation_m": held_translation,
                            "held_rotation_rad": held_rotation,
                        },
                        "score": {
                            "geometry_score": geometry_score,
                            "weighted_cost": weighted_cost,
                            "normalized_costs": normalized_costs,
                        },
                    }
                )

    candidates.sort(
        key=lambda candidate: (
            not candidate["eligible"],
            -candidate["score"]["geometry_score"],
            candidate["candidate_id"],
        )
    )
    return {
        "schema_version": 1,
        "kind": "button_goal_pose_candidates",
        "pure_geometry_only": True,
        "not_collision_or_reachability_certified": True,
        "source": {
            "button_center_radio_local": button_center_local.tolist(),
            "button_normal_radio_local": button_normal_local.tolist(),
            "press_position_world": press_position.tolist(),
            "press_direction_world": press_direction.tolist(),
            "head_optical_axis_world": (
                None if head_axis is None else head_axis.tolist()
            ),
            "transform_consistency_position_error_m": (consistency_position_error),
            "transform_consistency_orientation_error_rad": (
                consistency_orientation_error
            ),
        },
        "goal": {
            "chest_direction_world": chest_direction.tolist(),
            "chest_translation_m": chest_translation,
            "nominal_button_position_world": nominal_button_position.tolist(),
            "target_head_side_angle_deg": side_target_deg,
            "thresholds": {
                "max_face_to_press_angle_deg": face_limit_deg,
                "max_press_approach_opposition_angle_deg": (press_approach_limit_deg),
                "max_head_side_error_deg": side_limit_deg,
                "chest_translation_tolerance_m": chest_tolerance,
                "max_position_perturbation_m": max_position_perturbation,
                "max_orientation_perturbation_rad": (max_orientation_perturbation),
            },
            "score_weights": weights,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def generate_press_staging_pose_candidates(
    *,
    button_center_world: Any,
    button_normal_world: Any,
    world_press_transform: Any,
    standoff_m: float = 0.055,
    max_candidates: int = 8,
    alignment_phase: str = "final",
    eef_to_camera_transform: Any | None = None,
) -> dict[str, Any]:
    """Generate non-contact press EEF poses from measured button geometry.

    ``final`` aligns the press EEF local +Z axis with the button approach line
    at the strict 0.03--0.06 m pre-press standoff. ``observation`` instead
    aligns the real wrist-camera optical axis with the button at a farther
    non-contact camera standoff; the runtime derives the EEF pose through the
    live EEF-to-camera transform. In both phases roll is free for IK search.
    """

    center = np.asarray(button_center_world, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all():
        raise ValueError("button_center_world contains NaN or infinity")
    normal = _unit_vector(button_normal_world, name="button_normal_world")
    press = _rigid_transform(world_press_transform, name="world_press_transform")
    if alignment_phase not in {"final", "observation"}:
        raise ValueError("alignment_phase must be final or observation")
    standoff = float(standoff_m)
    lower = PREPRESS_AXIAL_STANDOFF_MIN_M
    upper = (
        PREPRESS_AXIAL_STANDOFF_MAX_M
        if alignment_phase == "final"
        else PRESS_STAGING_AXIAL_STANDOFF_MAX_M
    )
    if not math.isfinite(standoff) or not lower <= standoff <= upper:
        raise ValueError(
            f"standoff_m must lie within [{lower:.2f}, {upper:.2f}] "
            f"for alignment_phase={alignment_phase}"
        )
    if isinstance(max_candidates, bool) or not isinstance(
        max_candidates, (int, np.integer)
    ):
        raise ValueError("max_candidates must be an integer")
    budget = int(max_candidates)
    if not 1 <= budget <= 16:
        raise ValueError("max_candidates must lie within [1, 16]")

    current_quat = quaternion_xyzw_from_rotation_matrix(press[:3, :3])
    desired_direction = -normal
    eef_to_camera = None
    if alignment_phase == "observation":
        if eef_to_camera_transform is None:
            raise ValueError("observation alignment requires eef_to_camera_transform")
        eef_to_camera = _rigid_transform(
            eef_to_camera_transform, name="eef_to_camera_transform"
        )
        current_camera = press @ eef_to_camera
        current_camera_quat = quaternion_xyzw_from_rotation_matrix(
            current_camera[:3, :3]
        )
        current_optical_direction = _unit_vector(
            current_camera[:3, :3] @ np.asarray([0.0, 0.0, -1.0]),
            name="current_camera_optical_direction_world",
        )
        align_quat = quat_between_vectors_xyzw(
            current_optical_direction, desired_direction
        )
        aligned_quat = quat_multiply_xyzw(align_quat, current_camera_quat)
        target_origin = center + standoff * normal
    else:
        current_direction = _unit_vector(
            press[:3, :3] @ np.asarray([0.0, 0.0, 1.0]),
            name="current_press_direction_world",
        )
        align_quat = quat_between_vectors_xyzw(current_direction, desired_direction)
        aligned_quat = quat_multiply_xyzw(align_quat, current_quat)
        target_origin = center + standoff * normal
    roll_degrees = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0)
    candidates = []
    for index, roll_deg in enumerate(roll_degrees[:budget]):
        roll_quat, _ = _axis_angle_quaternion_xyzw(
            [*desired_direction.tolist(), math.radians(roll_deg)]
        )
        aligned_roll_quat = quat_multiply_xyzw(roll_quat, aligned_quat)
        aligned_transform = pose_matrix_xyzw(target_origin, aligned_roll_quat)
        if alignment_phase == "observation":
            assert eef_to_camera is not None
            target_press = aligned_transform @ np.linalg.inv(eef_to_camera)
            target_position = target_press[:3, 3]
            target_quat = quaternion_xyzw_from_rotation_matrix(target_press[:3, :3])
            predicted_camera = target_press @ eef_to_camera
            camera_to_button = center - predicted_camera[:3, 3]
            camera_optical = _unit_vector(
                predicted_camera[:3, :3] @ np.asarray([0.0, 0.0, -1.0]),
                name="predicted_camera_optical_direction_world",
            )
            camera_center_angle_deg = _vector_angle_deg(
                camera_optical, camera_to_button
            )
        else:
            target_position = target_origin
            target_quat = aligned_roll_quat
            camera_center_angle_deg = None
        target_rotation = pose_matrix_xyzw([0.0, 0.0, 0.0], target_quat)[:3, :3]
        actual_direction = _unit_vector(
            target_rotation @ np.asarray([0.0, 0.0, 1.0]),
            name="candidate_press_direction_world",
        )
        final_prepress_geometry = evaluate_geometry(
            button_center_world=center,
            button_normal_world=normal,
            press_eef_position_world=target_position,
            press_direction_world=actual_direction,
        )
        eligible = (
            final_prepress_geometry["geometry_pass"]
            if alignment_phase == "final"
            else bool(
                camera_center_angle_deg is not None and camera_center_angle_deg <= 1e-6
            )
        )
        candidates.append(
            {
                "candidate_id": (
                    f"press_{alignment_phase}_staging_roll_{roll_deg:+.0f}deg"
                ),
                "eligible": bool(eligible),
                "alignment_phase": alignment_phase,
                "roll_deg": roll_deg,
                "target_press_eef_pose": {
                    "position": target_position.tolist(),
                    "quat_xyzw": target_quat.tolist(),
                },
                "geometry": final_prepress_geometry,
                "final_prepress_geometry_pass": bool(
                    final_prepress_geometry["geometry_pass"]
                ),
                "predicted_button_camera_center_angle_deg": camera_center_angle_deg,
                "rotation_from_current_rad": quaternion_angle_rad(
                    current_quat, target_quat
                ),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            not candidate["eligible"],
            candidate["rotation_from_current_rad"],
            candidate["candidate_id"],
        )
    )
    return {
        "schema_version": 1,
        "kind": "press_staging_pose_candidates",
        "pure_geometry_only": True,
        "not_collision_or_reachability_certified": True,
        "button_center_world": center.tolist(),
        "button_normal_world": normal.tolist(),
        "standoff_m": standoff,
        "alignment_phase": alignment_phase,
        "standoff_reference": (
            "wrist_camera_origin" if alignment_phase == "observation" else "press_eef"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def quaternion_angle_rad(first: Any, second: Any) -> float:
    """Return the shortest angular distance between two xyzw quaternions."""

    left = np.asarray(first, dtype=np.float64).reshape(4)
    right = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("quaternion contains NaN or infinity")
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        raise ValueError("quaternion has zero length")
    dot = abs(float(np.dot(left / left_norm, right / right_norm)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def authorize_prepress_motion(
    *,
    role: str,
    current_xyz: Any,
    current_quat_xyzw: Any,
    target_xyz: Any,
    target_quat_xyzw: Any,
    face_class: str | None,
    direct_back_alignment: dict[str, Any] | None = None,
    press_observation_rotation: bool = False,
) -> dict[str, Any]:
    """Fail closed before planning motions that exceed the current visual cue."""

    current = np.asarray(current_xyz, dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("pre-press position contains NaN or infinity")
    translation = float(np.linalg.norm(target - current))
    rotation = quaternion_angle_rad(current_quat_xyzw, target_quat_xyzw)
    z_increase = float(target[2] - current[2])

    policy = "uncertain_bounded_search"
    translation_limit = UNCERTAIN_SEARCH_MAX_TRANSLATION_M
    rotation_limit = UNCERTAIN_SEARCH_MAX_ROTATION_RAD
    allowed = translation <= translation_limit and rotation <= rotation_limit

    if press_observation_rotation:
        if role != "press":
            raise ValueError(
                "press_observation_rotation is valid only for role='press'"
            )
        policy = "press_in_place_observation_rotation"
        translation_limit = PRESS_OBSERVATION_ROTATION_MAX_TRANSLATION_M
        rotation_limit = PRESS_OBSERVATION_ROTATION_MAX_ROTATION_RAD
        allowed = bool(
            translation <= translation_limit and rotation <= rotation_limit + 1e-9
        )
    elif role == "press" and face_class == BUTTON_FACE_CLASS:
        policy = "press_noncontact_staging"
        translation_limit = PRESS_STAGING_MAX_TRANSLATION_M
        rotation_limit = PRESS_STAGING_MAX_ROTATION_RAD
        allowed = bool(
            translation <= translation_limit + 1e-9
            and rotation <= rotation_limit + 1e-9
        )
    elif role == "press":
        policy = "press_requires_positive_button_projection"
        translation_limit = 0.0
        rotation_limit = 0.0
        allowed = False
    elif (
        role == "held"
        and z_increase >= SAFETY_RAISE_MIN_Z_M
        and translation <= SAFETY_RAISE_MAX_TRANSLATION_M
        and rotation <= UNCERTAIN_SEARCH_MAX_ROTATION_RAD
    ):
        policy = "safe_clearance_raise"
        translation_limit = SAFETY_RAISE_MAX_TRANSLATION_M
        rotation_limit = UNCERTAIN_SEARCH_MAX_ROTATION_RAD
        allowed = True
    elif role == "held" and face_class == CLEAR_SLOTTED_BACK_FACE_CLASS:
        policy = "direct_back_to_front"
        translation_limit = DIRECT_BACK_FACE_MAX_TRANSLATION_M
        rotation_limit = DIRECT_BACK_FACE_MAX_ROTATION_RAD
        allowed = bool(
            translation <= translation_limit + 1e-9
            and rotation <= rotation_limit + 1e-9
            and isinstance(direct_back_alignment, dict)
            and direct_back_alignment.get("valid") is True
        )
    elif role == "held" and face_class == BUTTON_FACE_CLASS:
        policy = "button_visible_micro_adjust"
        translation_limit = FINAL_MICRO_ADJUST_MAX_TRANSLATION_M
        rotation_limit = FINAL_MICRO_ADJUST_MAX_ROTATION_RAD
        allowed = translation <= translation_limit and rotation <= rotation_limit

    return {
        "allowed": bool(allowed),
        "policy": policy,
        "role": role,
        "face_class": face_class or AMBIGUOUS_FACE_CLASS,
        "translation_m": translation,
        "rotation_rad": rotation,
        "z_increase_m": z_increase,
        "translation_limit_m": translation_limit,
        "rotation_limit_rad": rotation_limit,
        "requires_plan_only_first": True,
        "direct_back_alignment": direct_back_alignment,
    }


def validate_button_declaration(
    *,
    button_visible: bool,
    positive_signature: dict[str, Any] | None,
    negative_case: str | None,
    bbox_xyxy: Any | None,
    center_uv: Any | None,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Validate the two-stage button gate without inventing pixel coordinates."""

    if not button_visible:
        if negative_case not in NEGATIVE_CASE_TO_FACE_CLASS:
            allowed = ", ".join(sorted(NEGATIVE_CASE_TO_FACE_CLASS))
            raise ValueError(f"NOT_VISIBLE negative_case must be one of: {allowed}")
        if (
            positive_signature is not None
            or bbox_xyxy is not None
            or center_uv is not None
        ):
            raise ValueError(
                "NOT_VISIBLE declaration must not include a positive signature "
                "or button coordinates"
            )
        return {
            "button_visible": False,
            "verdict": "NOT_VISIBLE",
            "face_class": NEGATIVE_CASE_TO_FACE_CLASS[negative_case],
            "positive_signature_complete": False,
            "negative_case": negative_case,
            "bbox_xyxy": None,
            "center_uv": None,
            "button_projection_authorized": False,
            "direct_held_pose_inference_authorized": (
                negative_case == "clear_slotted_back_face"
            ),
        }
    signature = positive_signature if isinstance(positive_signature, dict) else {}
    complete = all(signature.get(field) is True for field in POSITIVE_SIGNATURE_FIELDS)
    if not complete or negative_case:
        if bbox_xyxy is not None or center_uv is not None:
            raise ValueError("failed button signature must not include coordinates")
        return {
            "button_visible": False,
            "verdict": "NOT_VISIBLE",
            "face_class": AMBIGUOUS_FACE_CLASS,
            "positive_signature_complete": False,
            "negative_case": "ambiguous",
            "bbox_xyxy": None,
            "center_uv": None,
            "button_projection_authorized": False,
            "direct_held_pose_inference_authorized": False,
        }
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    center = np.asarray(center_uv, dtype=np.float64).reshape(2)
    if not np.isfinite(bbox).all() or not np.isfinite(center).all():
        raise ValueError("button pixel declaration is non-finite")
    x0, y0, x1, y1 = bbox.tolist()
    u, v = center.tolist()
    if not (0 <= x0 < x1 < image_width and 0 <= y0 < y1 < image_height):
        raise ValueError("button bbox is outside the declared frame")
    if not (x0 <= u <= x1 and y0 <= v <= y1):
        raise ValueError("button center is outside its bbox")
    return {
        "button_visible": True,
        "verdict": "VISIBLE",
        "face_class": BUTTON_FACE_CLASS,
        "positive_signature_complete": True,
        "positive_signature": dict.fromkeys(POSITIVE_SIGNATURE_FIELDS, True),
        "negative_case": None,
        "bbox_xyxy": [float(value) for value in bbox],
        "center_uv": [float(value) for value in center],
        "button_projection_authorized": True,
        "direct_held_pose_inference_authorized": True,
    }


def gate_token(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "button_gate_" + hashlib.sha256(encoded).hexdigest()[:20]


def evaluate_geometry(
    *,
    button_center_world: Any,
    button_normal_world: Any,
    press_eef_position_world: Any,
    press_direction_world: Any,
    max_line_distance_m: float = PREPRESS_LINE_DISTANCE_MAX_M,
    max_opposition_angle_deg: float = PREPRESS_OPPOSITION_ANGLE_MAX_DEG,
    min_axial_standoff_m: float = PREPRESS_AXIAL_STANDOFF_MIN_M,
    max_axial_standoff_m: float = PREPRESS_AXIAL_STANDOFF_MAX_M,
) -> dict[str, Any]:
    """Measure whether the button is ready for a straight local +Z press."""

    button = np.asarray(button_center_world, dtype=np.float64).reshape(3)
    press = np.asarray(press_eef_position_world, dtype=np.float64).reshape(3)
    if not np.isfinite(button).all() or not np.isfinite(press).all():
        raise ValueError("button or press-hand position is non-finite")
    normal = _unit_vector(button_normal_world, name="button_normal_world")
    direction = _unit_vector(press_direction_world, name="press_direction_world")
    offset = button - press
    axial = float(np.dot(offset, direction))
    radial = offset - axial * direction
    line_distance = float(np.linalg.norm(radial))
    normal_press_angle = math.degrees(
        math.acos(float(np.clip(np.dot(normal, direction), -1.0, 1.0)))
    )
    opposition_angle = math.degrees(
        math.acos(float(np.clip(np.dot(normal, -direction), -1.0, 1.0)))
    )
    face_toward_press = float(np.dot(normal, press - button)) > 0.0
    criteria = {
        "line_distance": line_distance <= float(max_line_distance_m) + 1e-9,
        "normal_opposition": opposition_angle <= float(max_opposition_angle_deg) + 1e-9,
        "axial_standoff": float(min_axial_standoff_m) - 1e-9
        <= axial
        <= float(max_axial_standoff_m) + 1e-9,
        "face_toward_press_hand": face_toward_press,
    }
    desired_axial = 0.5 * (float(min_axial_standoff_m) + float(max_axial_standoff_m))
    desired_button = press + desired_axial * direction
    return {
        "geometry_pass": all(criteria.values()),
        "criteria": criteria,
        "button_center_to_press_approach_line_m": line_distance,
        "button_normal_to_press_direction_deg": normal_press_angle,
        "button_normal_opposition_angle_deg": opposition_angle,
        "axial_standoff_m": axial,
        "face_toward_press_hand": face_toward_press,
        "press_direction_world": direction.tolist(),
        "desired_button_center_world": desired_button.tolist(),
        "suggested_button_translation_world": (desired_button - button).tolist(),
        "thresholds": {
            "max_line_distance_m": float(max_line_distance_m),
            "max_opposition_angle_deg": float(max_opposition_angle_deg),
            "min_axial_standoff_m": float(min_axial_standoff_m),
            "max_axial_standoff_m": float(max_axial_standoff_m),
        },
    }
