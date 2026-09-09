# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Real RLinf runtime dispatch with fake transports; no CAN or camera startup."""

import time

import numpy as np
import pytest

from robots.yam.rlinf_env import YamAgentEnv


@pytest.fixture
def dispatch_env(monkeypatch):
    backend_module = pytest.importorskip("rlinf.envs.realworld.yam.i2rt_backend")
    types = pytest.importorskip("rlinf.envs.realworld.yam.types")
    env = YamAgentEnv({"max_joint_delta_per_step": 0.02, "max_tracking_error_rad": 0.1})
    followers = []

    class Follower:
        def __init__(self):
            self.measured = np.array([0.08, 0.12, 0.2, 0, 0, 0, 0.5])
            self.commands = []

        def connect(self):
            pass

        def close(self):
            pass

        def hold(self):
            pass

        def assert_healthy(self, max_feedback_age_s):
            pass

        def read_state(self):
            return types.YamArmState(self.measured[:6], self.measured[6], time.time())

        def joint_limits(self):
            return np.column_stack((env.lower[0], env.upper[0]))

        def command(self, target):
            self.commands.append(target.copy())

    class Factory:
        def create_follower(self, device):
            follower = Follower()
            followers.append(follower)
            return follower

    monkeypatch.setattr(backend_module, "I2RTYamBackendFactory", Factory)
    env._runtime = env._build_runtime()
    env._runtime.connect_followers()
    env._previous_command = np.tile([0.10, 0.12, 0.2, 0, 0, 0, 0.5], 2)
    checked = []

    def check(start, target):
        checked.append((start.copy(), target.copy()))
        return {"ok": True}

    monkeypatch.setattr(env.geometry, "check_qpos_transition", check)
    try:
        yield env, followers, checked
    finally:
        env.close()


def test_actual_runtime_preserves_checked_target_with_nonuniform_lag(dispatch_env):
    env, followers, checked = dispatch_env
    target = env._previous_command.copy()
    target[[0, 7]] = 0.12
    target[[1, 8]] = 0.13
    accepted, _ = env._prepare_action(target)
    env._check_measured_transition(accepted)
    result = env._runtime.command(accepted)
    assert not env._runtime.config.enforce_runtime_joint_limits
    assert result.rejection_reason is None
    np.testing.assert_array_equal(result.accepted, accepted)
    np.testing.assert_array_equal(
        np.concatenate([f.commands[-1] for f in followers]), accepted
    )
    np.testing.assert_array_equal(checked[-1][1], accepted)
    # The old downstream clip independently reduced only the first joint's
    # progress, producing a different path despite passing the RPent check.
    measured = checked[-1][0]
    np.testing.assert_allclose((accepted - measured)[[0, 1]], [0.04, 0.01])


@pytest.mark.parametrize("rejection", ["hard_limit", "tracking", "collision"])
def test_rpent_rejections_remain_before_runtime_dispatch(
    dispatch_env, monkeypatch, rejection
):
    env, followers, _ = dispatch_env
    target = env._previous_command.copy()
    if rejection == "hard_limit":
        target[0] = env.upper[0, 0] + 0.01
        with pytest.raises(ValueError, match="hard limits"):
            env._prepare_action(target)
    elif rejection == "tracking":
        target[0] = 0.3
        with pytest.raises(RuntimeError, match="tracking error"):
            env._check_measured_transition(target)
    else:
        monkeypatch.setattr(
            env.geometry,
            "check_qpos_transition",
            lambda start, end: {"ok": False, "reason": "collision_guard"},
        )
        with pytest.raises(RuntimeError, match="collision_guard"):
            env._check_measured_transition(target)
    assert all(not follower.commands for follower in followers)
