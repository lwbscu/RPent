from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
import threading
from queue import Queue
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from robots.behavior.env_server import BehaviorEnvFacade


class _DirectProcess:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.need_observation: list[bool] = []

    def step_env(self, action, *, need_obs):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.need_observation.append(bool(need_obs))
        observation = object() if need_obs else None
        return (
            observation,
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            np.asarray([False]),
            [{"done": {"success": False}}],
        )


def _pure_vla_facade() -> tuple[BehaviorEnvFacade, _DirectProcess]:
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    direct = _DirectProcess()
    wrapped = {
        "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["pick up the trash"],
    }
    facade._env = SimpleNamespace(
        _direct_process=direct,
        _wrap_obs=lambda _obs: wrapped,
    )
    facade._done = False
    facade._last_observation = {
        key: value[0] if isinstance(value, np.ndarray) else value[0]
        for key, value in wrapped.items()
    }
    facade._last_info = {"done": {"success": False}}
    facade._env_steps = 0
    facade._planner_video_interval_steps = 10_000
    facade._gripper_latch = {"left": 1.0, "right": 1.0}
    facade._official_success_latched = False
    facade._official_success_receipt = None
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda *_args, **_kwargs: None
    return facade, direct


def test_pure_vla_complete_chunk_executes_all_32_actions_and_returns_once():
    facade, direct = _pure_vla_facade()

    observation, _reward, terminated, truncated, info = (
        facade._step_action_chunk(
            np.zeros((32, 23), dtype=np.float32),
            observe_final=True,
            pi0_nav_pick=True,
        )
    )

    assert observation is not None
    assert terminated is False
    assert truncated is False
    assert len(direct.actions) == 32
    assert direct.need_observation[:-1] == [False] * 31
    assert direct.need_observation[-1] is True
    monitor = info["_rpent"]["pi0_nav_pick_monitor"]
    assert monitor["executed_steps"] == 32
    assert monitor["stop_reason"] == "chunk_complete"


def test_episode_mp4_is_finalized_and_decodable(tmp_path):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._video_path = tmp_path / "episode.mp4"
    facade._video_writer = None
    facade._video_error = None
    facade._video_sealed = False
    facade._video_frames = 0
    facade._video_source_shapes = {}
    facade._video_queue = Queue(maxsize=64)
    facade._video_queue_stop = object()
    facade._video_queue_closed = False
    facade._video_frames_dropped = 0
    facade._video_worker = threading.Thread(
        target=facade._video_writer_loop,
        name="behavior-mp4-writer-test",
        daemon=True,
    )
    facade._video_worker.start()
    facade._planner_video_interval_steps = 4
    observation = {
        "main_images": np.full((8, 8, 3), 32, dtype=np.uint8),
        "wrist_images": np.stack(
            [
                np.full((4, 4, 3), 96, dtype=np.uint8),
                np.full((4, 4, 3), 160, dtype=np.uint8),
            ],
            axis=0,
        ),
    }

    facade._append_video(observation)
    facade._append_video(observation)
    facade._finalize_video_segment()

    assert facade._video_error is None
    assert facade._video_path.is_file()
    assert facade._video_path.stat().st_size > 0
    reader = imageio.get_reader(facade._video_path)
    try:
        decoded = reader.get_data(0)
    finally:
        reader.close()
    assert decoded.shape[:2] == (16, 16)
    assert facade._video_frames == 2
