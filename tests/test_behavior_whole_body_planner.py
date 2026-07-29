from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
from robots.behavior.planner_executor import (
    WHOLE_BODY_ACTIVE_JOINT_NAMES,
    WHOLE_BODY_LOCKED_JOINT_NAMES,
)


def test_official_r1pro_whole_body_active_and_locked_joint_layout_is_preserved():
    assert WHOLE_BODY_ACTIVE_JOINT_NAMES == (
        "base_footprint_x_joint",
        "base_footprint_y_joint",
        "base_footprint_rz_joint",
        "torso_joint1",
        "torso_joint2",
        "torso_joint3",
        "torso_joint4",
        *(f"left_arm_joint{i}" for i in range(1, 8)),
        *(f"right_arm_joint{i}" for i in range(1, 8)),
    )
    assert WHOLE_BODY_LOCKED_JOINT_NAMES == (
        "base_footprint_z_joint",
        "base_footprint_rx_joint",
        "base_footprint_ry_joint",
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    )
    assert len(WHOLE_BODY_ACTIVE_JOINT_NAMES) == 21
    assert len(WHOLE_BODY_LOCKED_JOINT_NAMES) == 7
