# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Cached diagnostics never connect or write controls, including on fallback."""

import json
import threading
import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from robots.yam.diagnostics import read_control_diagnostics


class Mapper:
    def to_command_joint_pos_space(self, values):
        result = values.copy()
        result[-1] = (values[-1] - 2) / 4
        return result

    def to_command_joint_vel_space(self, values):
        result = values.copy()
        result[-1] /= 4
        return result


class Robot:
    def __init__(self):
        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._commands = SimpleNamespace(
            pos=np.array([0.1] * 6 + [3.0]),
            vel=np.array([0.0] * 6 + [0.8]),
            kp=np.array([10.0] * 7),
            kd=np.array([1.5] * 7),
            torques=np.zeros(7),
            indices=None,
        )
        self._joint_state = SimpleNamespace(timestamp=time.time())
        self.remapper = Mapper()
        self.writes = []

    def get_observations(self):
        with self._state_lock:
            return {
                "joint_pos": np.array([0.09] * 6),
                "joint_vel": np.array([0.001] * 6),
                "joint_eff": np.array([0.2] * 6),
                "gripper_pos": np.array([0.24]),
                "gripper_vel": np.array([0.02]),
                "gripper_eff": np.array([0.3]),
            }

    def get_robot_info(self):
        return {
            "kp": np.array([80.0] * 7),
            "kd": np.array([5.0] * 7),
            "gravity_comp_factor": np.ones(7),
            "use_coulomb_friction": False,
        }

    def get_motor_torques(self):
        return np.array([0.5] * 7)

    def command_joint_pos(self, values):
        self.writes.append(values)
        raise AssertionError("diagnostics must never send commands")


def runtime_with(robot):
    return SimpleNamespace(_followers=[SimpleNamespace(_robot=robot)] * 2)


def test_actual_command_is_separate_from_configured_gains_and_normalized():
    robot = Robot()
    report = read_control_diagnostics(runtime_with(robot))
    assert report["status"] == "available"
    arm = report["arms"]["left"]
    assert arm["active_command"]["target_qpos"] == [0.1] * 6 + [0.25]
    assert arm["active_command"]["target_velocity"][-1] == pytest.approx(0.2)
    assert arm["active_command"]["kp_motor_space"] == [10.0] * 7
    assert arm["configured_control"]["kp_motor_space"] == [80.0] * 7
    assert arm["observations"]["joint_position_rad"] == [0.09] * 6
    assert arm["feedback"]["timestamp_s"] == robot._joint_state.timestamp
    assert not robot.writes
    json.dumps(report, allow_nan=False)
    robot._commands.pos[:] = 9
    assert arm["active_command"]["target_qpos"] == [0.1] * 6 + [0.25]


def test_hold_target_replaces_old_target_without_faking_a_control_write():
    robot = Robot()
    first = read_control_diagnostics(runtime_with(robot))
    with robot._command_lock:
        robot._commands.pos[:6] = 0.09
        robot._commands.kp[:] = 0
    second = read_control_diagnostics(runtime_with(robot))
    assert first["arms"]["left"]["active_command"]["target_qpos"][:6] == [0.1] * 6
    assert second["arms"]["left"]["active_command"]["target_qpos"][:6] == [0.09] * 6
    assert second["arms"]["left"]["active_command"]["kp_motor_space"] == [0.0] * 7
    assert not robot.writes


@pytest.mark.parametrize(
    "runtime", [None, SimpleNamespace(), SimpleNamespace(_followers=None)]
)
def test_disconnected_runtime_does_not_start_hardware(runtime):
    report = read_control_diagnostics(runtime)
    assert report["status"] == "unsupported"


def test_missing_sdk_command_capability_is_explicit_and_does_not_invent_target():
    robot = Robot()
    del robot._commands
    report = read_control_diagnostics(runtime_with(robot))
    arm = report["arms"]["left"]
    assert arm["observations"]["status"] == "available"
    assert arm["active_command"]["status"] == "unsupported"
    assert "target_qpos" not in arm["active_command"]
    assert not robot.writes


def test_busy_lock_and_nonfinite_feedback_are_reported_without_control_writes():
    robot = Robot()
    robot._joint_state.timestamp = float("nan")
    with robot._command_lock:
        report = read_control_diagnostics(runtime_with(robot))
    arm = report["arms"]["left"]
    assert arm["active_command"]["status"] == "unsupported"
    assert arm["feedback"]["status"] == "unsupported"
    json.dumps(report, allow_nan=False)
    assert not robot.writes


def test_cached_public_observation_with_nonreentrant_sdk_lock_completes():
    # Public get_observations locks internally; a helper must not acquire that
    # non-reentrant lock around the method and deadlock the control owner.
    robot = Robot()
    reports = []
    thread = threading.Thread(
        target=lambda: reports.append(read_control_diagnostics(runtime_with(robot))),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert reports[0]["status"] == "available"


def test_installed_sdk_public_observation_and_reversed_gripper_mapping():
    sdk = pytest.importorskip("i2rt.robots.motor_chain_robot")
    utils = pytest.importorskip("i2rt.robots.utils")
    robot = Robot()
    # Bind the installed SDK reader without constructing MotorChainRobot,
    # which would create control resources and may calibrate the gripper.
    robot.get_observations = MethodType(sdk.MotorChainRobot.get_observations, robot)
    robot._gripper_index = 6
    robot.temp_record_flag = False
    robot._joint_state = SimpleNamespace(
        pos=np.array([0.09] * 6 + [0.4]),
        vel=np.array([0.001] * 6 + [0.02]),
        eff=np.array([0.2] * 6 + [0.3]),
        timestamp=time.time(),
    )
    robot.remapper = utils.JointMapper(
        total_dofs=7, index_range_map={6: np.array([2.0, -2.0])}
    )
    robot._commands.pos[-1] = 1.0
    robot._commands.vel[-1] = -0.8
    report = read_control_diagnostics(runtime_with(robot))
    arm = report["arms"]["left"]
    assert arm["observations"]["gripper_position_normalized"] == 0.4
    assert arm["active_command"]["target_qpos"][-1] == pytest.approx(0.25)
    assert arm["active_command"]["target_velocity"][-1] == pytest.approx(0.2)
    assert not robot.writes
