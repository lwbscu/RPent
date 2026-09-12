# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Repeated stop preserves the first successful hold without opening hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.rlinf_env import YamAgentEnv
from robots.yam.toolkit import YamToolkit


class Runtime:
    def __init__(self):
        self.measured = np.zeros(14)
        self.holds = []
        self.fail_hold = False
        self.closed = False
        self.command_entered = threading.Event()
        self.command_release = threading.Event()
        self.command_release.set()

    def read_state(self):
        return SimpleNamespace(as_vector=lambda: self.measured.copy())

    def hold(self):
        assert not self.closed
        if self.fail_hold:
            raise RuntimeError("hold failed")
        self.holds.append(self.measured.copy())
        return self.measured.copy()

    def command(self, target):
        assert not self.closed
        self.command_entered.set()
        assert self.command_release.wait(timeout=2)
        self.measured = target.copy()
        return SimpleNamespace(
            accepted=target.copy(), rejection_reason=None, clipped=False
        )

    def close(self):
        self.closed = True


@pytest.fixture
def env(monkeypatch):
    runtime = Runtime()
    env = YamAgentEnv({}, runtime=runtime, cameras=object())
    env._started = True
    env._previous_command = runtime.measured.copy()
    env._operator_ready_receipt = {"event": "ready"}
    monkeypatch.setattr(env, "_pace", lambda: None)
    monkeypatch.setattr(env.geometry, "robot_state", lambda q: {"qpos": q.tolist()})
    monkeypatch.setattr(
        env.geometry, "check_qpos_transition", lambda a, b: {"ok": True}
    )
    yield env
    runtime.command_release.set()
    env.close()
    worker = env._stop_worker
    if worker is not None:
        worker.join(timeout=2)
        assert not worker.is_alive()


def test_servo_stop_then_toolkit_close_does_not_recapture_sag(env):
    env.request_stop()
    stamp = env._last_stop_hold_s
    first_target = env._runtime.holds[0].copy()
    env._runtime.measured[3] -= 0.025
    # Exactly the real toolkit cleanup method, with fake primitive dependencies.
    toolkit = SimpleNamespace(
        _primitives=SimpleNamespace(env=env, stop_recording=lambda: []),
        _save_episode_video=lambda: None,
    )
    YamToolkit.close(toolkit)
    env.request_stop()
    assert len(env._runtime.holds) == 1
    np.testing.assert_array_equal(env._runtime.holds[0], first_target)
    assert env._last_stop_hold_s == stamp


def test_failed_hold_is_not_latched_and_later_stop_can_retry(env):
    env._runtime.fail_hold = True
    with pytest.raises(RuntimeError, match="hold failed"):
        env.request_stop()
    assert env._stop_hold_episode_id is None
    env._runtime.fail_hold = False
    env.request_stop()
    env.request_stop()
    assert len(env._runtime.holds) == 1


def test_stop_during_last_command_holds_once_without_another_step(env):
    runtime = env._runtime
    runtime.command_release.clear()
    errors = []

    def run():
        try:
            env.control_step(np.zeros(14), expected_episode_id=env._episode_id)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert runtime.command_entered.wait(timeout=1)
    env.request_stop()
    worker = env._stop_worker
    for _ in range(5):
        env.request_stop()
        assert env._stop_worker is worker
    runtime.command_release.set()
    thread.join(timeout=2)
    worker.join(timeout=2)
    assert not thread.is_alive() and not worker.is_alive()
    assert not errors
    assert len(runtime.holds) == 1


@pytest.mark.parametrize("transition", ["reset", "close"])
def test_deferred_old_stop_does_not_hold_new_episode_or_closed_runtime(
    env, monkeypatch, transition
):
    # Another thread owns the env lock when the request arrives. This thread
    # then completes reset/close before the queued stop can take that same lock.
    with env._lock:
        request = threading.Thread(target=env.request_stop)
        request.start()
        request.join(timeout=1)
        assert not request.is_alive()
        worker = env._stop_worker
        assert worker is not None
        if transition == "reset":
            monkeypatch.setattr(
                env, "_consume_ready_receipt_locked", lambda: {"event": "ready"}
            )
            monkeypatch.setattr(env, "_observe_locked", lambda: ({}, {}))
            env.reset()
        else:
            env.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not env._runtime.holds
    if transition == "reset":
        assert not env._stop_requested.is_set()
        env.request_stop()
        assert len(env._runtime.holds) == 1


def test_diagnostics_read_after_stop_does_not_write_another_hold(env):
    env.request_stop()
    env._runtime.measured[3] -= 0.02
    for _ in range(3):
        env.read_control_state()
    assert len(env._runtime.holds) == 1


def test_stop_after_successful_close_does_not_write_or_open_runtime(env):
    env.close()
    env.request_stop()
    assert not env._runtime.holds
