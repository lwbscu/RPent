# Copyright (c) 2026 RPent contributors

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_base_curobo_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "behavior_base_curobo_probe_oracles",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(step: int = 7):
    return {
        "env_steps": step,
        "joint_state": {"sha256": "unchanged"},
    }


class _Planner:
    def __init__(self):
        self.calls = []

    def _wrist_camera_rotation_calibration(self, hand):
        return {
            "verified": True,
            "hand": hand,
            "visual_ccw_angle_sign": 1.0,
            "screen_normal_axis_eef": (
                [0.0, 0.0, 1.0] if hand == "left" else [0.0, 1.0, 0.0]
            ),
        }

    def move_to(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "primitive_success": True,
            "task_success": False,
            "metrics": {
                "env_actions_sent": 0,
                "whole_body_certificate": {"collision_free": True},
            },
        }


class _Backend:
    def __init__(self):
        self.torso_calls = []

    def get_torso_pose(self):
        return np.asarray([1.0, 2.0, 0.9]), np.asarray([0.0, 0.0, 0.0, 1.0])

    def plan_torso_trajectory(self, **kwargs):
        self.torso_calls.append(kwargs)
        return {
            "ok": True,
            "joint_trajectory": np.zeros((2, 3), dtype=np.float32),
            "metrics": {
                "env_actions_sent": 0,
                "collision_admission": {"admitted": True},
            },
        }

    def get_eef_pose(self, hand):
        x = -0.4 if hand == "left" else 0.4
        return np.asarray([x, 0.2, 1.0]), np.asarray([0.0, 0.0, 0.0, 1.0])


def test_planning_only_torso_and_both_wrist_directions_are_zero_action(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    backend = _Backend()
    env = SimpleNamespace(
        _env_steps=7,
        _last_info={"done": {"success": False}},
        _official_success_latched=False,
        _planner=_Planner(),
    )
    monkeypatch.setattr(module, "_runtime_snapshot", lambda *_args: _snapshot())

    torso = [
        module._plan_only_torso_probe(
            env,
            backend,
            action=action,
        )
        for action in ("up", "down")
    ]
    wrist = [
        module._plan_only_wrist_probe(
            env,
            backend,
            hand=hand,
            action=action,
        )
        for hand in ("left", "right")
        for action in ("rotate_left", "rotate_right")
    ]

    assert [call["target_z_m"] for call in backend.torso_calls] == pytest.approx(
        [0.93, 0.87]
    )
    assert all(case["zero_action_verified"] for case in torso + wrist)
    assert all(case["plan_admitted"] for case in torso + wrist)
    assert all(case["release_admission"] is False for case in torso + wrist)
    assert all(call["plan_only"] is True for call in env._planner.calls)
    assert [case["requested_rotation_rad"] for case in wrist] == pytest.approx(
        [
            math.radians(5.0),
            -math.radians(5.0),
            math.radians(5.0),
            -math.radians(5.0),
        ]
    )
    assert wrist[0]["target_quaternion_xyzw"] != wrist[1]["target_quaternion_xyzw"]
    assert wrist[0]["target_quaternion_xyzw"] != wrist[2]["target_quaternion_xyzw"]


def test_planning_only_probe_stops_before_any_planner_call_on_raw_success(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()

    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected post-success call: {name}")

    env = SimpleNamespace(
        _last_info={"done": {"success": True}},
        _official_success_latched=True,
        _planner=NoCalls(),
    )
    monkeypatch.setattr(
        module,
        "_runtime_snapshot",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("snapshot after raw success")
        ),
    )

    torso = module._plan_only_torso_probe(env, NoCalls(), action="up")
    wrist = module._plan_only_wrist_probe(
        env,
        NoCalls(),
        hand="left",
        action="rotate_left",
    )

    for case in (torso, wrist):
        assert case["skipped"] is True
        assert case["skip_reason"] == "official_task_success_latched"
        assert case["no_follow_up_rpc"] is True
        assert case["env_step_delta"] == 0
