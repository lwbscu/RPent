# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy

import numpy as np
import pytest

from robots.yam.primitives import YamPrimitives
from robots.yam.servo import BoundedJointPI, JointServoConfig, run_joint_servo


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, dt):
            self.now += dt

    value = Clock()
    monkeypatch.setattr("robots.yam.servo.time.monotonic", value.monotonic)
    monkeypatch.setattr("robots.yam.servo.time.sleep", value.sleep)
    return value


class Plant:
    def __init__(self, clock, *, servo=None, frozen=False):
        self.clock = clock
        self.execution_capabilities = {"joint_servo": servo}
        self.command = np.r_[np.full(6, 0.4), 0.3, np.full(6, 0.5), 0.7]
        self.measured = self.command.copy()
        self.measured[:6] -= 0.01
        self.frozen = frozen
        self.stops = 0
        self.commands = []
        self.status = {
            "episode_id": "one",
            "take_action_cnt": 0,
            "step_lim": 1000,
            "eval_success": False,
            "stop_requested": False,
        }
        self.refresh()

    def refresh(self):
        if not self.frozen:
            self.measured = self.command.copy()
            self.measured[:6] -= 0.01
        self.last_obs = {
            "state": {"joint_position": self.measured.copy()},
            "frames": {},
        }
        self.last_info = {
            "commanded_qpos": self.command.copy(),
            "episode_status": self.status.copy(),
            "robot_state": {
                "left_eef_pose": np.r_[self.measured[:3], 1, 0, 0, 0],
                "right_eef_pose": np.r_[self.measured[7:10], 1, 0, 0, 0],
            },
        }

    def observe(self):
        self.clock.now += 0.001
        self.refresh()
        return self.last_obs, self.last_info

    def plan_arm_path(self, arm, target):
        return {
            "status": "Success",
            "position": np.array([np.full(6, 0.39), np.full(6, 0.4)]),
        }

    def chunk_step(self, actions, **kwargs):
        assert kwargs["expected_episode_id"] == "one"
        for action in actions:
            self.command = np.asarray(action).copy()
            self.commands.append(self.command.copy())
            self.status["take_action_cnt"] += 1
        self.refresh()
        info = deepcopy(self.last_info)
        info["executed_actions"] = len(actions)
        return {}, 0, False, False, info

    def request_stop(self):
        self.stops += 1
        self.status["stop_requested"] = True
        return {"stopped": True}


def run(plant, **kwargs):
    primitive = YamPrimitives(env=plant, check_cancelled=lambda: None)
    return run_joint_servo(
        env=plant,
        apply_updates=primitive.apply_qpos_updates,
        check_cancelled=kwargs.pop("check_cancelled", lambda: None),
        arm="left",
        nominal=np.full(6, 0.4),
        target_pose=np.array([0.4, 0.4, 0.4, 1, 0, 0, 0]),
        episode_id="one",
        config=JointServoConfig.from_config({"enabled": True, "ki": 1, **kwargs}),
    )


@pytest.mark.parametrize(
    "value",
    [
        {"foo": 1},
        {"enabled": 1},
        {"ki": True},
        {"kp": float("nan")},
        {"ki": 3},
        {"period_s": 0.01},
        {"max_bias_rad": 0.03},
        {"max_bias_step_rad": 0.003},
        {"stable_samples": True},
        {"stable_samples": 2},
        {"timeout_s": 4},
    ],
)
def test_config_rejects_unbounded_or_ambiguous_settings(value):
    with pytest.raises(ValueError):
        JointServoConfig.from_config(value)


def test_empty_configuration_disabled():
    assert not JointServoConfig.from_config(None).enabled
    assert not JointServoConfig.from_config({}).enabled


def test_pi_step_total_clips_and_antiwindup():
    controller = BoundedJointPI(
        JointServoConfig.from_config({"enabled": True, "ki": 2})
    )
    error = np.full(6, 0.1)
    previous = np.zeros(6)
    for _ in range(100):
        bias = controller.update(error, 100.0)
        assert np.max(abs(bias - previous)) <= 0.002 + 1e-12
        assert np.max(abs(bias)) <= 0.025
        previous = bias
    np.testing.assert_allclose(controller.integral, controller.bias)
    reversed_bias = controller.update(-error, 0.1)
    np.testing.assert_allclose(reversed_bias, 0.023)
    controller.accept(np.zeros(6), error)
    np.testing.assert_allclose(controller.integral, 0)


def test_success_retains_bias_and_other_arm_grippers(clock):
    plant = Plant(clock)
    untouched = plant.command[6:].copy()
    result = run(plant)
    assert result["success"], result
    assert plant.stops == 0
    assert len(result["trace"]) >= 3
    assert result["position_error_m"] <= 0.005
    assert result["max_joint_error_rad"] <= 0.008
    assert np.max(abs(plant.command[:6] - 0.4)) <= 0.025
    assert np.max(abs(plant.command[:6] - 0.4)) > 0
    for command in plant.commands:
        np.testing.assert_array_equal(command[6:], untouched)
    for entry in result["trace"][-3:]:
        assert entry["position_error_m"] <= 0.005


@pytest.mark.parametrize("timeout,reason", [(1.0, "no_progress"), (3.0, "timeout")])
def test_stationary_plant_stops_without_unbounded_windup(clock, timeout, reason):
    plant = Plant(clock, frozen=True)
    result = run(plant, progress_timeout_s=timeout)
    assert not result["success"]
    assert result["stop_reason"] == reason
    assert plant.stops == 1
    assert all(
        np.max(abs(command[:6] - 0.4)) <= 0.025 + 1e-12 for command in plant.commands
    )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("budget", "budget_exhausted"),
        ("episode", "episode_changed"),
        ("failure", "episode_stopped"),
        ("abort", "episode_stopped"),
        ("nan", "feedback_error"),
        ("range", "feedback_error"),
        ("cancel", "cancelled"),
    ],
)
def test_faults_stop_before_correction(clock, fault, reason):
    plant = Plant(clock, frozen=True)
    if fault == "budget":
        plant.status["step_lim"] = 0
    if fault == "episode":
        plant.status["episode_id"] = "two"
    if fault in ("failure", "abort"):
        plant.status["terminal_event"] = fault
    if fault == "nan":
        plant.measured[0] = np.nan
    if fault == "range":
        plant.measured[1] = -1

    def cancelled():
        if fault == "cancel":
            raise RuntimeError("cancelled")

    result = run(plant, check_cancelled=cancelled)
    assert not result["success"]
    assert result["stop_reason"] == reason
    assert plant.stops == 1
    assert plant.commands == []


def test_move_to_servo_has_strict_pose_criterion_and_combined_counts(clock):
    plant = Plant(clock, servo={"enabled": True, "ki": 1})
    primitive = YamPrimitives(env=plant, check_cancelled=lambda: None)
    result = primitive.move_to(arm="left", xyz=[0.4, 0.4, 0.4], quat=[1, 0, 0, 0])
    assert result["success"], result
    assert result["position_tolerance_m"] == 0.005
    assert result["path_executed_steps"] == 2
    assert result["servo_executed_steps"] > 0
    assert result["executed_steps"] == len(plant.commands)
    assert result["executed_steps"] == 2 + result["servo_executed_steps"]


def test_disabled_retains_legacy_completion(clock):
    plant = Plant(clock)
    result = YamPrimitives(env=plant, check_cancelled=lambda: None).move_to(
        arm="left", xyz=[0.4, 0.4, 0.4]
    )
    assert result["success"]
    assert result["position_tolerance_m"] == 0.025
    assert "servo" not in result
    assert len(plant.commands) == 2


def test_servo_does_not_accept_legacy_17mm_pose_error(clock):
    plant = Plant(clock, servo={"enabled": True}, frozen=True)
    result = YamPrimitives(env=plant, check_cancelled=lambda: None).move_to(
        arm="left", xyz=[0.4, 0.4, 0.4]
    )
    assert not result["success"]
    assert result["position_error_m"] > 0.005
    assert plant.stops == 1


def test_downstream_clipping_back_calculates_integral(clock):
    plant = Plant(clock)
    original = plant.chunk_step

    def clipped(actions, **kwargs):
        bounded = np.asarray(actions).copy()
        bounded[:, :6] = np.minimum(bounded[:, :6], 0.401)
        return original(bounded, **kwargs)

    plant.chunk_step = clipped
    result = run(plant)
    assert not result["success"]
    assert result["stop_reason"] == "no_progress"
    assert np.max(np.abs(result["final_bias_rad"])) <= 0.001 + 1e-12
    assert (
        max(max(entry.get("requested_bias_rad", [0])) for entry in result["trace"])
        < 0.0021
    )
    assert plant.stops == 1


def test_previous_corrected_command_does_not_replace_fixed_nominal(clock):
    plant = Plant(clock)
    plant.command[:6] += 0.01
    result = run(plant)
    assert not result["success"]
    assert result["stop_reason"] == "command_mismatch"
    assert not plant.commands
    assert plant.stops == 1


class CompactPlant(Plant):
    def __init__(self, clock):
        super().__init__(clock, servo={"enabled": True, "ki": 1})
        self.execution_capabilities["compact_control"] = True
        self.last_control_obs = None
        self.last_control_info = None
        self.compact_reads = 0
        self.compact_steps = 0

    def observe(self):
        raise AssertionError("PI must not fetch RGBD with compact control available")

    def cache_control(self):
        self.last_control_obs = deepcopy(self.last_obs)
        self.last_control_info = deepcopy(self.last_info)
        self.last_control_info["control_diagnostics"] = {
            "arms": {
                "left": {
                    "observations": {
                        "joint_position_rad": self.measured[:6],
                        "joint_velocity_rad_s": np.zeros(6),
                        "joint_effort_nm": np.ones(6),
                    },
                    "feedforward": {"motor_torque_nm": np.full(7, 0.2)},
                    "active_command": {
                        "target_qpos": self.command[:7],
                        "kp_motor_space": np.full(7, 80),
                        "kd_motor_space": np.full(7, 5),
                    },
                },
                "right": {"active_command": {"target_qpos": np.full(7, 99)}},
            }
        }

    def read_control_state(self):
        old_obs, old_info = self.last_obs, self.last_info
        self.clock.now += 0.001
        self.refresh()
        self.cache_control()
        self.last_obs, self.last_info = old_obs, old_info
        self.compact_reads += 1
        return self.last_control_obs, self.last_control_info

    def control_step(self, action, **kwargs):
        old_obs, old_info = self.last_obs, self.last_info
        _, reward, terminated, truncated, info = super().chunk_step([action], **kwargs)
        self.cache_control()
        self.last_control_info["executed_actions"] = info["executed_actions"]
        self.last_obs, self.last_info = old_obs, old_info
        self.compact_steps += 1
        return (
            self.last_control_obs,
            reward,
            terminated,
            truncated,
            self.last_control_info,
        )


def test_compact_pi_uses_separate_cache_and_traces_existing_motor_diagnostics(clock):
    plant = CompactPlant(clock)
    untouched = plant.command[6:].copy()
    primitive = YamPrimitives(env=plant, check_cancelled=lambda: None)
    result = primitive.move_to(arm="left", xyz=[0.4, 0.4, 0.4], quat=[1, 0, 0, 0])
    assert result["success"], result
    assert plant.compact_reads >= 3
    assert plant.compact_steps == result["servo_executed_steps"] > 0
    assert (
        result["requested_actions"] == result["executed_actions"] == len(plant.commands)
    )
    assert result["episode_status"]["take_action_cnt"] == len(plant.commands)
    assert plant.last_info["episode_status"]["take_action_cnt"] == 2
    assert result["servo"]["compact_control"]
    for entry in result["servo"]["trace"]:
        assert len(entry["driver"]["target_qpos"]) == 6
        assert entry["driver"]["kp_motor_space"] == [80] * 6
        assert entry["driver"]["kd_motor_space"] == [5] * 6
        assert entry["driver"]["joint_effort_nm"] == [1] * 6
        assert entry["driver"]["feedforward_motor_torque_nm"] == [0.2] * 6
        assert entry["driver"]["joint_velocity_rad_s"] == [0] * 6
        assert entry["feedback_read_s"] == pytest.approx(0.001)
    for command in plant.commands:
        np.testing.assert_array_equal(command[6:], untouched)


def test_compact_updates_reject_multi_step_before_dispatch(clock):
    plant = CompactPlant(clock)
    primitive = YamPrimitives(env=plant, check_cancelled=lambda: None)
    with pytest.raises(ValueError, match="exactly one"):
        primitive.apply_qpos_updates(
            [{"arm": "left", "arm_qpos": [0.4] * 6}] * 2, compact_control=True
        )
    assert not plant.commands
