from __future__ import annotations

from types import SimpleNamespace

import pytest

from robots.behavior.env_server import BehaviorEnvFacade


class _FrameCache:
    def __init__(self):
        self.group = "capture:4:old"
        self.step = 4
        self.images = {
            "head": b"head-old",
            "left_wrist": b"left-old",
            "right_wrist": b"right-old",
        }

    def latest(self, camera):
        return SimpleNamespace(capture_group_id=self.group)

    def observe_payload(self, camera):
        return {
            "frame_id": f"{camera}:{self.step}:{self.group}",
            "capture_group": {
                "id": self.group,
                "sim_step": self.step,
            },
            "_image_bytes": self.images[camera],
        }


def test_dashboard_capture_returns_one_fresh_three_camera_group():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 5
    facade._frame_cache = _FrameCache()

    def refresh():
        facade._frame_cache.group = "capture:5:new"
        facade._frame_cache.step = 5
        facade._frame_cache.images = {
            "head": b"head-new",
            "left_wrist": b"left-new",
            "right_wrist": b"right-new",
        }

    facade._refresh_observation_without_step = refresh

    result = facade._dashboard_capture_group()

    assert result["_frames_bytes"] == {
        "head": b"head-new",
        "left_wrist": b"left-new",
        "right_wrist": b"right-new",
    }
    assert result["capture_group_id"] == "capture:5:new"
    assert result["simulator_step"] == 5
    assert set(result["frame_ids"]) == {"head", "left_wrist", "right_wrist"}


def test_dashboard_capture_rejects_nonfresh_group_instead_of_republishing_old_frames():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 4
    facade._frame_cache = _FrameCache()
    old_images = dict(facade._frame_cache.images)
    facade._refresh_observation_without_step = lambda: None

    with pytest.raises(RuntimeError, match="fresh capture group"):
        facade._dashboard_capture_group()

    assert facade._frame_cache.images == old_images
    assert facade._frame_cache.group == "capture:4:old"


def test_dashboard_capture_rejects_partial_camera_payload():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 5
    facade._frame_cache = _FrameCache()

    def refresh():
        facade._frame_cache.group = "capture:5:bad"
        facade._frame_cache.step = 5
        facade._frame_cache.images["right_wrist"] = None

    facade._refresh_observation_without_step = refresh

    with pytest.raises(RuntimeError, match="right_wrist"):
        facade._dashboard_capture_group()


def test_dashboard_capture_rejects_mixed_lineage():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 5
    cache = _FrameCache()
    facade._frame_cache = cache

    def refresh():
        cache.group = "capture:5:new"
        cache.step = 5

        original = cache.observe_payload

        def mixed(camera):
            payload = original(camera)
            if camera == "right_wrist":
                payload["capture_group"] = {
                    "id": "capture:5:other",
                    "sim_step": 5,
                }
            return payload

        cache.observe_payload = mixed

    facade._refresh_observation_without_step = refresh

    with pytest.raises(RuntimeError, match="one capture group"):
        facade._dashboard_capture_group()
