import types

import numpy as np
import pytest

from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.schemas import PI0_PICK_VLA_MODE


def _raw_proprio(hand: str, opening: float) -> np.ndarray:
    raw = np.zeros(256, dtype=np.float32)
    segment = slice(193, 195) if hand == "left" else slice(232, 234)
    raw[segment] = float(opening) / 2.0
    return raw


class _Robot:
    gripper_control_idx = {"left": [24, 25], "right": [26, 27]}

    def __init__(self):
        self.q = np.zeros(28, dtype=np.float64)

    def get_joint_positions(self):
        return self.q.copy()


class _DirectProcess:
    def __init__(self, robot, openings, *, hand="right", public_offset=0.0):
        self.robot = robot
        self.openings = list(openings)
        self.hand = hand
        self.public_offset = float(public_offset)
        self.calls = 0
        self.current_raw = _raw_proprio(hand, 0.1)

    def step_env(self, _action, *, need_obs):
        opening = float(self.openings[self.calls])
        self.calls += 1
        indices = self.robot.gripper_control_idx[self.hand]
        self.robot.q[indices] = opening / 2.0
        self.current_raw = _raw_proprio(
            self.hand, opening + self.public_offset
        )
        raw_obs = [{"states": self.current_raw.copy()}] if need_obs else None
        info = {"done": {"success": False, "termination_conditions": {}}}
        return raw_obs, np.array([0.0]), np.array([False]), np.array([False]), [info]


class _Env:
    def __init__(self, direct, robot):
        self._direct_process = direct
        self.robots = [robot]

    @staticmethod
    def _wrap_obs(raw_observations):
        raw = raw_observations[0]["states"]
        return {
            "main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
            "states": np.asarray([raw], dtype=np.float32),
            "task_descriptions": ["local grasp"],
        }


def _facade(openings, *, hand="right", public_offset=0.0):
    robot = _Robot()
    direct = _DirectProcess(
        robot, openings, hand=hand, public_offset=public_offset
    )
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env = _Env(direct, robot)
    facade._control_mode = PI0_PICK_VLA_MODE
    facade._done = False
    facade._env_steps = 0
    facade._last_observation = None
    facade._last_info = None
    facade._planner_video_interval_steps = 4
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade.video_append_calls = 0

    def append_video(_observation):
        facade.video_append_calls += 1

    facade._append_video = append_video
    facade.refresh_calls = 0

    def refresh_without_step(self):
        self.refresh_calls += 1
        wrapped = self._env._wrap_obs(
            [{"states": direct.current_raw.copy()}]
        )
        self._last_observation = {
            "main_images": wrapped["main_images"][0],
            "wrist_images": wrapped["wrist_images"][0],
            "states": wrapped["states"][0],
            "task_descriptions": wrapped["task_descriptions"][0],
        }

    facade._refresh_observation_without_step = types.MethodType(
        refresh_without_step, facade
    )
    return facade, direct


def test_pi0_monitor_stops_after_three_physical_closed_steps_and_refreshes():
    facade, direct = _facade([0.04, 0.04, 0.04, 0.1])

    observation, _reward, terminated, truncated, info = facade.pi0_chunk_step(
        np.zeros((4, 23), dtype=np.float32),
        hand="right",
        gripper_closed_threshold=0.045,
        required_closed_steps=3,
        stop_on_candidate=True,
    )

    monitor = info["_rpent"]["local_gripper_monitor"]
    assert set(monitor) == {
        "hand",
        "opening",
        "min_opening",
        "closed_streak",
        "candidate",
        "candidate_env_step",
        "executed_steps",
    }
    assert monitor == {
        "hand": "right",
        "opening": pytest.approx(0.04),
        "min_opening": pytest.approx(0.04),
        "closed_streak": 3,
        "candidate": True,
        "candidate_env_step": 3,
        "executed_steps": 3,
    }
    assert direct.calls == 3
    assert facade.refresh_calls == 1
    assert facade.video_append_calls == 1
    assert observation["states"][232:234].sum() == pytest.approx(0.04)
    assert terminated is False
    assert truncated is False


def test_pi0_monitor_records_candidate_without_stopping_when_not_requested():
    facade, direct = _facade([0.04, 0.04, 0.04, 0.1])

    *_rest, info = facade.pi0_chunk_step(
        np.zeros((4, 23), dtype=np.float32),
        hand="right",
        gripper_closed_threshold=0.045,
        stop_on_candidate=False,
    )

    monitor = info["_rpent"]["local_gripper_monitor"]
    assert direct.calls == 4
    assert facade.refresh_calls == 0
    assert monitor["candidate"] is True
    assert monitor["candidate_env_step"] == 3
    assert monitor["closed_streak"] == 0
    assert monitor["opening"] == pytest.approx(0.1)
    assert monitor["min_opening"] == pytest.approx(0.04)
    assert monitor["executed_steps"] == 4


def test_pi0_monitor_uses_configured_consecutive_step_requirement():
    facade, direct = _facade([0.04, 0.04, 0.1])

    *_rest, info = facade.pi0_chunk_step(
        np.zeros((3, 23), dtype=np.float32),
        hand="right",
        required_closed_steps=2,
        stop_on_candidate=True,
    )

    monitor = info["_rpent"]["local_gripper_monitor"]
    assert direct.calls == 2
    assert monitor["closed_streak"] == 2
    assert monitor["candidate"] is True
    assert monitor["candidate_env_step"] == 2


def test_pi0_candidate_refresh_does_not_duplicate_interval_video_frame():
    facade, direct = _facade([0.04, 0.04, 0.04, 0.04])

    facade.pi0_chunk_step(
        np.zeros((4, 23), dtype=np.float32),
        hand="right",
        required_closed_steps=4,
        stop_on_candidate=True,
    )

    assert direct.calls == 4
    assert facade.refresh_calls == 1
    assert facade.video_append_calls == 1


def test_pi0_monitor_uses_selected_left_physical_and_public_joints():
    facade, direct = _facade([0.03], hand="left")

    *_rest, info = facade.pi0_chunk_step(
        np.zeros((1, 23), dtype=np.float32),
        hand="left",
        gripper_closed_threshold=0.045,
        stop_on_candidate=False,
    )

    monitor = info["_rpent"]["local_gripper_monitor"]
    assert direct.calls == 1
    assert monitor["hand"] == "left"
    assert monitor["opening"] == pytest.approx(0.03)
    assert monitor["closed_streak"] == 1
    assert monitor["candidate"] is False


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_pi0_monitor_rejects_invalid_required_closed_steps(value):
    facade, direct = _facade([0.03])

    with pytest.raises(ValueError, match="required_closed_steps"):
        facade.pi0_chunk_step(
            np.zeros((1, 23), dtype=np.float32),
            hand="right",
            required_closed_steps=value,
        )

    assert direct.calls == 0


def test_pi0_monitor_fails_closed_on_physical_public_opening_mismatch():
    facade, direct = _facade([0.04, 0.04], public_offset=0.02)

    with pytest.raises(
        RuntimeError, match="same-step physical/public gripper opening mismatch"
    ):
        facade.pi0_chunk_step(
            np.zeros((2, 23), dtype=np.float32),
            hand="right",
            gripper_closed_threshold=0.045,
            stop_on_candidate=True,
        )

    assert direct.calls == 1
