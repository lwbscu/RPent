# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.yam.contracts import (
    MODEL_SPEC,
    env_runtime_contract,
    validate_actions,
)
from robots.yam.env_client import YamEnvClient
from robots.yam.env_server import YamEnvFacade
from robots.yam.hardware_ownership import HardwareLease
from robots.yam.operator_control import write_receipt
from robots.yam.primitives import YamPrimitives
from robots.yam.rlinf_env import YamAgentEnv
from robots.yam.toolkit import YamToolkit
from robots.yam.vla_server import YamVLAFacade, build_model_cfg
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.robots.components.vla_client_base import BaseVLAClient
from rpent.utils.rpc.http_rpc import HttpRpcClient
from rpent.utils.rpc.rpc_client import RpcError
from rpent.utils.rpc.socket_rpc import SocketRpcClient


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _served_facade(facade: Any, transport: str) -> Iterator[Any]:
    port = _free_port()
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            facade.serve(transport=transport, host="127.0.0.1", port=port)
        except BaseException as exc:  # pragma: no cover - surfaced by caller.
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    if transport == "http":
        client = HttpRpcClient(f"http://127.0.0.1:{port}")
    else:
        client = SocketRpcClient("127.0.0.1", port)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if errors:
            raise errors[0]
        try:
            assert client.call("healthz", timeout_s=0.2) == {"status": "ok"}
            break
        except Exception:
            time.sleep(0.02)
    else:
        raise TimeoutError(f"{transport} facade did not become ready")

    try:
        yield client
    finally:
        try:
            client.call("shutdown", timeout_s=1.0)
        except Exception:
            pass
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        if errors:
            raise errors[0]


class FakeYamEnv:
    def __init__(self, *, block_chunk: bool = False) -> None:
        self.block_chunk = block_chunk
        self.reset_calls = 0
        self.step_actions: list[np.ndarray] = []
        self.chunk_actions: list[np.ndarray] = []
        self.stop_calls = 0
        self.close_calls = 0
        self.chunk_entered = threading.Event()
        self.stop_requested = threading.Event()

    def get_task_language(self) -> str:
        return "place the cube"

    def observe(self) -> dict[str, Any]:
        return self._observation(), self._info()

    def _observation(self) -> dict[str, Any]:
        return {
            "frames": {
                "top": np.full((3, 4, 3), 11, dtype=np.uint8),
                "left": np.full((3, 4, 3), 22, dtype=np.uint8),
                "right": np.full((3, 4, 3), 33, dtype=np.uint8),
            },
            "state": {"joint_position": np.arange(14, dtype=np.float32)},
        }

    def _info(self) -> dict[str, Any]:
        return {
            "episode_status": {
                "eval_success": False,
                "take_action_cnt": len(self.step_actions)
                + sum(len(actions) for actions in self.chunk_actions),
                "step_lim": 40,
                "actual_seed": 12,
                "episode_id": "fake-yam-episode",
            },
            "robot_state": {"qpos": np.arange(14, dtype=np.float64)},
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.reset_calls += 1
        obs = self._observation()
        obs["seed"] = seed
        obs["options"] = options or {}
        obs["reset_calls"] = self.reset_calls
        return obs, self._info()

    def step(
        self,
        action: np.ndarray,
        *,
        action_type: str = "qpos",
        expected_episode_id: str | None = None,
    ):
        assert expected_episode_id in {None, "fake-yam-episode"}
        flat = validate_actions(action, action_type=action_type)
        if flat.shape[0] != 1:
            raise ValueError("YAM step requires one qpos14 action")
        self.step_actions.append(flat[0].copy())
        return (
            self._observation(),
            0.0,
            False,
            False,
            {**self._info(), "executed_actions": 1, "action_type": "qpos"},
        )

    def chunk_step(
        self,
        actions: np.ndarray,
        *,
        action_type: str = "qpos",
        return_all_frames: bool = False,
        expected_episode_id: str | None = None,
    ):
        assert expected_episode_id in {None, "fake-yam-episode"}
        actions = validate_actions(actions, action_type=action_type)
        self.chunk_actions.append(actions.copy())
        if self.block_chunk:
            self.chunk_entered.set()
            if not self.stop_requested.wait(timeout=2.0):
                raise TimeoutError("chunk_step was not stopped")
        frames = [self._observation() | {"frame": i} for i in range(len(actions))]
        obs = frames if return_all_frames else frames[-1]
        return (
            obs,
            np.zeros(len(actions), dtype=np.float32),
            False,
            False,
            {
                **self._info(),
                "action_type": action_type,
                "executed_actions": int(len(actions)),
                "stop_calls": self.stop_calls,
            },
        )

    def render_camera(self, camera_name: str, *, depth: bool = False):
        value = {"top": 11, "left": 22, "right": 33}[camera_name]
        rgb = np.full((3, 4, 3), value, dtype=np.uint8)
        if not depth:
            return rgb
        return rgb, np.full((3, 4), value / 100.0, dtype=np.float32)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return {"name": camera_name, "intrinsic_K": np.eye(3, dtype=np.float32)}

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict[str, Any]:
        return {"arm": arm, "target_pose": np.asarray(target_pose, dtype=np.float32)}

    def request_stop(self) -> None:
        self.stop_calls += 1
        self.stop_requested.set()

    def close(self) -> None:
        self.close_calls += 1


class FakeYamModel:
    def __init__(self, actions: np.ndarray | None = None) -> None:
        if actions is None:
            actions = np.zeros((1, MODEL_SPEC.action_horizon, 14), dtype=np.float32)
            actions[..., 6] = 0.25
            actions[..., 13] = 0.75
        self.actions = actions
        self.calls: list[tuple[dict[str, Any], str]] = []

    def predict_action_batch(self, env_obs: dict[str, Any], *, mode: str):
        self.calls.append((env_obs, mode))
        return self.actions.copy(), {"source": "fake"}


class FakeToolkitModel:
    def __init__(self, on_predict: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.on_predict = on_predict

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        self.calls.append(obs)
        if self.on_predict is not None:
            self.on_predict()
        return _valid_action_chunk(MODEL_SPEC.use_length)[None]


class FakeRpc:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple, dict]] = []

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        del timeout_s
        self.calls.append((method, args, kwargs or {}))
        result = self.responses[method]
        return result() if callable(result) else result


class FakeRobotState:
    def __init__(self, qpos: np.ndarray) -> None:
        self.qpos = qpos
        self.left = SimpleNamespace(timestamp_s=10.0)
        self.right = SimpleNamespace(timestamp_s=11.0)

    def as_vector(self) -> np.ndarray:
        return self.qpos.copy()


class FakeRuntime:
    def __init__(
        self,
        qpos: np.ndarray | None = None,
        *,
        accepted_updates_measured_state: bool = False,
        on_command: Any = None,
    ) -> None:
        if qpos is None:
            qpos = np.zeros(14, dtype=np.float64)
            qpos[[6, 13]] = 0.5
        self.qpos = qpos
        self.accepted_updates_measured_state = accepted_updates_measured_state
        self.on_command = on_command
        self.connect_calls = 0
        self.hold_calls = 0
        self.close_calls = 0
        self.commands: list[np.ndarray] = []

    def connect_followers(self) -> None:
        self.connect_calls += 1

    def hold(self) -> None:
        self.hold_calls += 1

    def read_state(self) -> FakeRobotState:
        return FakeRobotState(self.qpos)

    def command(self, action: np.ndarray):
        accepted = np.asarray(action, dtype=np.float64).copy()
        self.commands.append(accepted)
        if self.accepted_updates_measured_state:
            self.qpos = accepted.copy()
        if self.on_command is not None:
            self.on_command()
        return SimpleNamespace(
            accepted=accepted,
            clipped=False,
            rejection_reason=None,
        )

    def close(self) -> None:
        self.close_calls += 1


class FakeCameraRig:
    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.snapshot_count = 0

    def open(self) -> None:
        self.open_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def snapshot(
        self, *, not_before_monotonic_s: float | None = None
    ) -> dict[str, Any]:
        del not_before_monotonic_s
        self.snapshot_count += 1
        before = time.monotonic()
        after = time.monotonic()
        views = {}
        for index, name in enumerate(("top", "left", "right"), start=1):
            views[name] = SimpleNamespace(
                rgb=np.full((2, 3, 3), index, dtype=np.uint8),
                depth=np.full((2, 3), float(index), dtype=np.float32),
                camera_meta={
                    "name": name,
                    "intrinsic_K": np.eye(3, dtype=np.float64),
                    "cam2world_cv": np.eye(4, dtype=np.float64),
                    "timestamps": {
                        "host_before_monotonic_s": before,
                        "host_after_monotonic_s": after,
                    },
                },
            )
        return {"snapshot_id": f"s{self.snapshot_count}", "views": views}

    def render_camera(self, camera_name: str, *, depth: bool = False):
        frame = self.snapshot()["views"][camera_name]
        return (frame.rgb, frame.depth) if depth else frame.rgb

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self.snapshot()["views"][camera_name].camera_meta


class FreshSnapshotCameraRig:
    def __init__(
        self,
        *,
        bracketed: bool = True,
        include_timestamps: bool = True,
    ) -> None:
        self.bracketed = bracketed
        self.include_timestamps = include_timestamps
        self.open_calls = 0
        self.close_calls = 0
        self.snapshot_not_before_calls: list[float | None] = []

    def open(self) -> None:
        self.open_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def snapshot(
        self, *, not_before_monotonic_s: float | None = None
    ) -> dict[str, Any]:
        self.snapshot_not_before_calls.append(not_before_monotonic_s)
        before = time.monotonic()
        after = time.monotonic()
        if not self.bracketed and not_before_monotonic_s is not None:
            before = max(0.0, float(not_before_monotonic_s) - 1e-6)
            after = time.monotonic()
        views = {}
        for index, name in enumerate(("top", "left", "right"), start=1):
            camera_meta = {
                "name": name,
                "width": 3,
                "height": 2,
                "intrinsic_K": np.eye(3, dtype=np.float64),
                "cam2world_cv": np.eye(4, dtype=np.float64),
            }
            if self.include_timestamps:
                camera_meta["timestamps"] = {
                    "host_before_monotonic_s": before,
                    "host_after_monotonic_s": after,
                }
            views[name] = SimpleNamespace(
                rgb=np.full((2, 3, 3), index, dtype=np.uint8),
                depth=np.full((2, 3), float(index), dtype=np.float32),
                camera_meta=camera_meta,
            )
        return {
            "snapshot_id": "fresh-fake",
            "views": views,
            "capture_host_interval_s": (before, after),
            "sync_mode": "fresh_fake",
        }

    def render_camera(self, camera_name: str, *, depth: bool = False):
        frame = self.snapshot()["views"][camera_name]
        return (frame.rgb, frame.depth) if depth else frame.rgb

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self.snapshot()["views"][camera_name].camera_meta


class MovingReadRuntime(FakeRuntime):
    def __init__(self, before: np.ndarray, after: np.ndarray) -> None:
        super().__init__(before)
        self._states = [before.copy(), before.copy(), after.copy()]
        self.read_count = 0

    def read_state(self) -> FakeRobotState:
        index = min(self.read_count, len(self._states) - 1)
        self.read_count += 1
        return FakeRobotState(self._states[index])


class FakeKinematics:
    def fk(self, joints: np.ndarray, gripper: float) -> np.ndarray:
        del gripper
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(joints[0])
        return transform


class FakeToolkitEnv:
    def __init__(
        self,
        *,
        success_after_chunk: bool = False,
        require_expected_episode_id: bool = False,
    ) -> None:
        self.server_meta = {"task_name": "place_cube"}
        self.execution_capabilities = {"chunk_step_all_frames": True}
        self.success_after_chunk = success_after_chunk
        self.require_expected_episode_id = require_expected_episode_id
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 20
        self.actual_seed = 12
        self.episode_id = "tk-episode-0"
        self.qpos = np.zeros(14, dtype=np.float64)
        self.qpos[[6, 13]] = 0.5
        self.stop_requests: list[str] = []
        self.hold_calls = 0
        self.reset_calls = 0
        self.chunk_step_calls: list[dict[str, Any]] = []
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None
        self.observe()

    def force_new_episode_id(self) -> None:
        self.episode_id = (
            f"tk-episode-{self.reset_calls + len(self.chunk_step_calls) + 1}"
        )
        self.eval_success = False
        self.take_action_cnt = 0
        self.observe()

    def _status(self) -> dict[str, Any]:
        return {
            "eval_success": self.eval_success,
            "take_action_cnt": self.take_action_cnt,
            "step_lim": self.step_lim,
            "actual_seed": self.actual_seed,
            "episode_id": self.episode_id,
        }

    def _observation(self) -> dict[str, Any]:
        views = {}
        for value, name in enumerate(("top", "left", "right"), start=1):
            rgb = np.full((3, 4, 3), value * 20, dtype=np.uint8)
            depth = np.full((3, 4), 1.0 + value, dtype=np.float32)
            views[name] = {
                "rgb": rgb,
                "depth": depth,
                "camera_meta": {
                    "name": name,
                    "width": 4,
                    "height": 3,
                    "intrinsic_K": np.eye(3, dtype=np.float64),
                    "cam2world_cv": np.eye(4, dtype=np.float64),
                },
            }
        return {
            "frames": {name: view["rgb"] for name, view in views.items()},
            "state": {"joint_position": self.qpos.copy()},
            "views": views,
            "snapshot_id": f"tk-{self.take_action_cnt}",
        }

    def _info(self) -> dict[str, Any]:
        return {
            "episode_status": self._status(),
            "robot_state": {
                "qpos": self.qpos.copy(),
                "left_eef_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
                "right_eef_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
                "world_frame": "left_base",
            },
        }

    def observe(self):
        self.last_obs = self._observation()
        self.last_info = self._info()
        return self.last_obs, self.last_info

    def reset(self, *, reason: str = ""):
        del reason
        self.reset_calls += 1
        self.eval_success = False
        self.take_action_cnt = 0
        self.episode_id = f"tk-reset-{self.reset_calls}"
        return self.observe()

    def chunk_step(
        self,
        actions: np.ndarray,
        *,
        action_type: str = "qpos",
        return_all_frames: bool = False,
        expected_episode_id: str | None = None,
    ):
        del action_type
        actions = validate_actions(actions)
        self.chunk_step_calls.append({"expected_episode_id": expected_episode_id})
        if self.require_expected_episode_id and expected_episode_id is None:
            raise AssertionError("pi05_act did not bind expected_episode_id")
        if expected_episode_id is not None and expected_episode_id != self.episode_id:
            self.hold_calls += 1
            raise RuntimeError("stale episode_id refused before command")
        self.take_action_cnt += len(actions)
        if len(actions):
            self.qpos = actions[-1].astype(np.float64, copy=True)
        if self.success_after_chunk:
            self.eval_success = True
        obs, info = self.observe()
        payload = [obs for _ in range(len(actions))] if return_all_frames else obs
        return (
            payload,
            1.0 if self.eval_success else 0.0,
            self.eval_success,
            False,
            {**info, "executed_actions": len(actions)},
        )

    def step(
        self,
        action: np.ndarray,
        *,
        action_type: str = "qpos",
        expected_episode_id: str | None = None,
    ):
        return self.chunk_step(
            validate_actions(action),
            action_type=action_type,
            return_all_frames=False,
            expected_episode_id=expected_episode_id,
        )

    def plan_arm_path(self, arm: str, target_pose: Any) -> dict[str, Any]:
        del target_pose
        return {"status": "Success", "position": np.zeros((2, 6)), "arm": arm}

    def get_task_language(self) -> str:
        return "place the cube"

    def request_stop(self, reason: str = "") -> dict[str, Any]:
        self.stop_requests.append(reason)
        return {"stop_requested": True}


def _yam_agent_config(
    *,
    tmp_path: Path,
    step_lim: int = 40,
    operator_receipt_path: Path | None = None,
    include_wrist_handeye: bool = True,
) -> dict[str, Any]:
    identity = np.eye(4, dtype=np.float64).tolist()
    extrinsics: dict[str, Any] = {
        "world_frame": "left_base",
        "top_camera": {
            "T_base_to_cam": {
                "via_left_arm": identity,
                "via_right_arm": identity,
            }
        },
    }
    if include_wrist_handeye:
        extrinsics["left_wrist"] = {"T_grasp_to_cam": identity}
        extrinsics["right_wrist"] = {"T_grasp_to_cam": identity}
    extrinsics_path = tmp_path / "solve_handeye-extrinsics.json"
    extrinsics_path.write_text(json.dumps(extrinsics), encoding="utf-8")

    return {
        "task_name": "place_cube",
        "task_language": "place the cube",
        "seed": 12,
        "max_episode_steps": step_lim,
        "max_joint_delta_per_step": None,
        "operator_receipt_path": (
            None if operator_receipt_path is None else str(operator_receipt_path)
        ),
        "table_z": -1.0,
        "table_clearance_m": 0.01,
        "extrinsics_path": str(extrinsics_path),
    }


def _write_operator_receipt(env: Any, path: Path, *, event: str) -> dict[str, Any]:
    info = getattr(env, "last_info", None)
    if not isinstance(info, dict):
        _, info = env.observe()
    status = info["episode_status"]
    return write_receipt(
        path,
        episode_id=status["episode_id"],
        event=event,
        note=f"test operator {event}",
    )


def _observe_write_ready_reset(env: Any, path: Path):
    _, pending_info = env.observe()
    pending_id = pending_info["episode_status"]["episode_id"]
    receipt = _write_operator_receipt(env, path, event="ready")
    assert receipt["episode_id"] == pending_id
    obs, info = env.reset()
    ready = info["episode_status"]["operator_ready_receipt"]
    assert info["episode_status"]["episode_id"] != pending_id
    assert ready["reset_request_episode_id"] == pending_id
    assert ready["ready_for_episode_id"] == info["episode_status"]["episode_id"]
    return obs, info


def _valid_action_chunk(length: int = 2) -> np.ndarray:
    actions = np.zeros((length, 14), dtype=np.float64)
    actions[:, 6] = 0.5
    actions[:, 13] = 1.0
    return actions


def _valid_vla_observation() -> dict[str, Any]:
    top = np.full((1, 3, 4, 3), 10, dtype=np.uint8)
    left = np.full((3, 4, 3), 20, dtype=np.uint8)
    right = np.full((3, 4, 3), 30, dtype=np.uint8)
    side = np.stack([left, right])[None]
    states = np.arange(14, dtype=np.float32)[None]
    return {
        "main_images": top,
        "extra_view_images": side,
        "states": states,
        "task_descriptions": ["place the cube"],
    }


_DEFAULT_TOOLKIT_MODEL = object()


def test_yam_env_facade_constructor_does_not_reset_or_move_robot() -> None:
    env = FakeYamEnv()

    facade = YamEnvFacade(env)

    assert facade.get_env_meta()["runtime"] == "yam_real_env"
    obs, _ = facade.observe()
    assert obs["state"]["joint_position"].shape == (14,)
    assert env.reset_calls == 0
    assert env.step_actions == []
    assert env.chunk_actions == []
    assert env.stop_calls == 0


def test_yam_env_client_constructor_observes_without_resetting() -> None:
    env = FakeYamEnv()
    expected_meta = env_runtime_contract(
        task_name="place_cube",
        seed=12,
        max_episode_steps=40,
    )
    rpc = FakeRpc(
        {
            "env.get_env_meta": expected_meta,
            "env.observe": env.observe,
            "env.reset": env.reset,
        }
    )

    client = YamEnvClient(rpc, expected_meta=expected_meta)

    assert client.last_obs["state"]["joint_position"].shape == (14,)
    assert env.reset_calls == 0
    assert [call[0] for call in rpc.calls] == ["env.get_env_meta", "env.observe"]


def test_common_env_client_default_still_resets_on_connect() -> None:
    env = FakeYamEnv()
    expected_meta = {"runtime": "generic"}
    rpc = FakeRpc(
        {
            "env.get_env_meta": expected_meta,
            "env.reset": env.reset,
        }
    )

    client = BaseEnvClient(rpc, expected_meta=expected_meta)

    assert client.last_obs[0]["reset_calls"] == 1
    assert env.reset_calls == 1
    assert [call[0] for call in rpc.calls] == ["env.get_env_meta", "env.reset"]


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_yam_env_facade_localhost_rpc_round_trip_and_cleanup(transport: str) -> None:
    env = FakeYamEnv()
    facade = YamEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name="place_cube",
            seed=12,
            max_episode_steps=40,
        ),
    )
    actions = _valid_action_chunk(3)

    with _served_facade(facade, transport) as client:
        assert client.call("env.get_env_meta")["seed"] == 12
        reset_obs, reset_info = client.call(
            "env.reset",
            kwargs={"seed": 99, "options": {"operator_ready": True}},
        )
        assert reset_obs["seed"] == 99
        assert reset_info["episode_status"]["actual_seed"] == 12
        step = client.call("env.step", args=(actions[0],))
        chunk = client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": True},
        )
        rgb, depth = client.call(
            "env.render_camera",
            kwargs={"camera_name": "left", "depth": True},
        )
        meta = client.call("env.get_camera_meta", args=("right",))

    assert step[4]["executed_actions"] == 1
    assert chunk[4]["executed_actions"] == 3
    assert len(chunk[0]) == 3
    assert np.array_equal(env.step_actions[0], actions[0])
    assert np.array_equal(env.chunk_actions[0], actions)
    assert rgb.shape == (3, 4, 3)
    assert depth.shape == (3, 4)
    assert meta["name"] == "right"
    assert env.close_calls == 1


@pytest.mark.parametrize(
    ("actions", "action_type"),
    [
        (np.zeros((0, 14), dtype=np.float64), "qpos"),
        (np.zeros((1, 13), dtype=np.float64), "qpos"),
        (np.zeros((1, 15), dtype=np.float64), "qpos"),
        (np.full((1, 14), np.nan, dtype=np.float64), "qpos"),
        (np.array([[0, 0, 0, 0, 0, 0, -0.1, 0, 0, 0, 0, 0, 0, 0.5]]), "qpos"),
        (np.array([[0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 1.1]]), "qpos"),
        (_valid_action_chunk(), "delta"),
    ],
)
def test_yam_action_contract_rejects_invalid_shape_nan_grip_and_delta(
    actions: np.ndarray,
    action_type: str,
) -> None:
    with pytest.raises(ValueError):
        validate_actions(actions, action_type=action_type)


def test_yam_env_facade_rejects_bad_actions_before_calling_runtime() -> None:
    env = FakeYamEnv()
    facade = YamEnvFacade(env)

    with pytest.raises(ValueError):
        facade.step(np.zeros((2, 14), dtype=np.float64))
    with pytest.raises(ValueError):
        facade.chunk_step(np.zeros((1, 13), dtype=np.float64))

    assert env.step_actions == []
    assert env.chunk_actions == []


def test_yam_env_client_reset_round_trips_without_constructor_auto_reset() -> None:
    env = FakeYamEnv()
    expected_meta = env_runtime_contract(
        task_name="place_cube",
        seed=12,
        max_episode_steps=40,
    )
    facade = YamEnvFacade(env, metadata=expected_meta)

    with _served_facade(facade, "http") as rpc:
        client = YamEnvClient(rpc, expected_meta=expected_meta)
        assert env.reset_calls == 0
        obs, info = client.reset()

    assert obs["state"]["joint_position"].shape == (14,)
    assert info["episode_status"]["actual_seed"] == 12
    assert env.reset_calls == 1


def test_request_stop_bypasses_writer_lock_during_active_chunk() -> None:
    env = FakeYamEnv(block_chunk=True)
    facade = YamEnvFacade(env)
    chunk_errors: list[BaseException] = []
    chunk_result: list[Any] = []

    with _served_facade(facade, "http") as client:

        def run_chunk() -> None:
            try:
                chunk_result.append(
                    client.call(
                        "env.chunk_step",
                        args=(_valid_action_chunk(4),),
                        timeout_s=5.0,
                    )
                )
            except BaseException as exc:
                chunk_errors.append(exc)

        thread = threading.Thread(target=run_chunk)
        thread.start()
        assert env.chunk_entered.wait(timeout=1.0)

        started = time.monotonic()
        assert client.call("env.request_stop", timeout_s=1.0) == {
            "stop_requested": True,
            "hold_confirmed": False,
        }
        elapsed = time.monotonic() - started
        thread.join(timeout=2.0)

    assert elapsed < 0.5
    assert not thread.is_alive()
    assert chunk_errors == []
    assert chunk_result[0][4]["executed_actions"] == 4
    assert env.stop_calls == 1


def test_yam_agent_env_missing_handeye_still_returns_rgb_and_status(tmp_path) -> None:
    env = YamAgentEnv(
        _yam_agent_config(
            tmp_path=tmp_path,
            operator_receipt_path=tmp_path / "operator-receipt.json",
            include_wrist_handeye=False,
        ),
        runtime=FakeRuntime(),
        cameras=FreshSnapshotCameraRig(bracketed=True),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        obs, info = env.observe()
    finally:
        env.close()

    assert set(obs["frames"]) == {"top", "left", "right"}
    assert info["episode_status"]["episode_id"]
    for name in ("left", "right"):
        meta = obs["views"][name]["camera_meta"]
        assert meta["projection_valid"] is False
        assert "missing_or_invalid_wrist_handeye" in meta["projection_limitation"]


def test_yam_agent_env_unbracketed_cached_snapshot_invalidates_wrist_projection(
    tmp_path,
) -> None:
    config = _yam_agent_config(
        tmp_path=tmp_path,
        operator_receipt_path=tmp_path / "operator-receipt.json",
    )
    config["projection_observe_timeout_s"] = 0.0
    env = YamAgentEnv(
        config,
        runtime=FakeRuntime(),
        cameras=FreshSnapshotCameraRig(bracketed=False),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        obs, _ = env.observe()
    finally:
        env.close()

    for name in ("left", "right"):
        meta = obs["views"][name]["camera_meta"]
        assert meta["projection_valid"] is False
        assert "not bracketed" in meta["projection_limitation"]


def test_yam_agent_env_top_projection_does_not_depend_on_stationary_qpos(
    tmp_path,
) -> None:
    before = np.zeros(14, dtype=np.float64)
    before[[6, 13]] = 0.5
    after = before.copy()
    after[0] += 0.02
    config = _yam_agent_config(
        tmp_path=tmp_path,
        operator_receipt_path=tmp_path / "operator-receipt.json",
    )
    config["qpos_static_tolerance_rad"] = 1e-4
    config["projection_observe_timeout_s"] = 0.0
    env = YamAgentEnv(
        config,
        runtime=MovingReadRuntime(before, after),
        cameras=FreshSnapshotCameraRig(bracketed=True),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        obs, _ = env.observe()
    finally:
        env.close()

    assert obs["views"]["top"]["camera_meta"]["projection_valid"] is True
    for name in ("left", "right"):
        meta = obs["views"][name]["camera_meta"]
        assert meta["projection_valid"] is False
        assert "joints moved" in meta["projection_limitation"]


def test_yam_agent_env_observe_waits_fresh_but_step_uses_cached_snapshot(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    cameras = FreshSnapshotCameraRig(bracketed=True)
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=FakeRuntime(accepted_updates_measured_state=True),
        cameras=cameras,
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        env.observe()
        assert any(call is not None for call in cameras.snapshot_not_before_calls)
        _write_operator_receipt(env, receipt_path, event="ready")
        env.reset()
        cameras.snapshot_not_before_calls.clear()
        env.step(_valid_action_chunk(1)[0])
    finally:
        env.close()

    assert cameras.snapshot_not_before_calls
    assert cameras.snapshot_not_before_calls == [None, None]


def test_yam_agent_env_stop_during_pace_holds_without_commanding(tmp_path) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    runtime = FakeRuntime(accepted_updates_measured_state=True)
    config = _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path)
    config["projection_observe_timeout_s"] = 0.0
    env = YamAgentEnv(
        config,
        runtime=runtime,
        cameras=FreshSnapshotCameraRig(bracketed=True),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    errors: list[BaseException] = []
    result: list[Any] = []
    try:
        _observe_write_ready_reset(env, receipt_path)
        startup_holds = runtime.hold_calls
        env._next_tick_s = time.perf_counter() + 0.25

        def run_chunk() -> None:
            try:
                result.append(env.chunk_step(_valid_action_chunk(1)))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run_chunk)
        thread.start()
        time.sleep(0.05)
        env.request_stop()
        thread.join(timeout=1.0)
    finally:
        env.close()

    assert not thread.is_alive()
    assert errors == []
    assert runtime.commands == []
    assert runtime.hold_calls > startup_holds
    assert result[0][4]["executed_actions"] == 0
    assert result[0][3] is True
    assert result[0][4]["episode_status"]["stop_requested"] is True


def test_yam_agent_env_observe_does_not_fabricate_operator_ready_receipt(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=FakeRuntime(),
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        _, info = env.observe()
    finally:
        env.close()

    assert info["episode_status"]["ready_for_motion"] is False
    assert info["episode_status"]["operator_ready_receipt"] is None


def test_yam_agent_env_reset_without_ready_holds_and_keeps_episode_id(tmp_path) -> None:
    runtime = FakeRuntime()
    env = YamAgentEnv(
        _yam_agent_config(
            tmp_path=tmp_path,
            operator_receipt_path=tmp_path / "operator-receipt.json",
        ),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        _, info = env.observe()
        pending_id = info["episode_status"]["episode_id"]
        startup_holds = runtime.hold_calls
        with pytest.raises(
            RuntimeError, match="requires a local operator ready receipt"
        ):
            env.reset()
        _, after = env.observe()
    finally:
        env.close()

    assert after["episode_status"]["episode_id"] == pending_id
    assert after["episode_status"]["ready_for_motion"] is False
    assert runtime.hold_calls > startup_holds


def test_yam_agent_env_ready_reset_starts_new_episode_id(tmp_path) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=FakeRuntime(),
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        _, pending_info = env.observe()
        pending_id = pending_info["episode_status"]["episode_id"]
        _write_operator_receipt(env, receipt_path, event="ready")
        _, reset_info = env.reset()
    finally:
        env.close()

    new_id = reset_info["episode_status"]["episode_id"]
    ready = reset_info["episode_status"]["operator_ready_receipt"]
    assert new_id != pending_id
    assert reset_info["episode_status"]["ready_for_motion"] is True
    assert ready["reset_request_episode_id"] == pending_id
    assert ready["ready_for_episode_id"] == new_id


def test_yam_agent_env_stale_success_receipt_is_not_reused_after_reset(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=FakeRuntime(),
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        _, pending_info = env.observe()
        stale_id = pending_info["episode_status"]["episode_id"]
        _write_operator_receipt(env, receipt_path, event="ready")
        _, reset_info = env.reset()
        current_id = reset_info["episode_status"]["episode_id"]
        write_receipt(
            receipt_path,
            episode_id=stale_id,
            event="success",
            note="old success must not apply to new episode",
        )
        _, after = env.observe()
    finally:
        env.close()

    assert current_id != stale_id
    assert after["episode_status"]["episode_id"] == current_id
    assert after["episode_status"]["eval_success"] is False
    assert after["episode_status"]["terminal_event"] is None
    assert after["episode_status"]["operator_success_receipt"] is None


def test_yam_agent_env_rejects_stale_expected_episode_without_commanding(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    runtime = FakeRuntime(accepted_updates_measured_state=True)
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        _, pending_info = env.observe()
        stale_id = pending_info["episode_status"]["episode_id"]
        _write_operator_receipt(env, receipt_path, event="ready")
        _, reset_info = env.reset()
        assert reset_info["episode_status"]["episode_id"] != stale_id
        startup_holds = runtime.hold_calls
        with pytest.raises(RuntimeError, match="episode"):
            env.chunk_step(_valid_action_chunk(1), expected_episode_id=stale_id)
    finally:
        env.close()

    assert runtime.commands == []
    assert runtime.hold_calls > startup_holds


def test_yam_agent_env_requires_operator_ready_before_motion(tmp_path) -> None:
    runtime = FakeRuntime()
    env = YamAgentEnv(
        _yam_agent_config(
            tmp_path=tmp_path,
            operator_receipt_path=tmp_path / "operator-receipt.json",
        ),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        env.observe()
        startup_holds = runtime.hold_calls
        with pytest.raises(RuntimeError, match="operator ready receipt"):
            env.chunk_step(_valid_action_chunk(1))
    finally:
        env.close()

    assert runtime.hold_calls > startup_holds
    assert runtime.commands == []


def test_yam_agent_env_rejects_agent_supplied_operator_receipt_options(
    tmp_path,
) -> None:
    env = YamAgentEnv(
        _yam_agent_config(
            tmp_path=tmp_path,
            operator_receipt_path=tmp_path / "operator-receipt.json",
        ),
        runtime=FakeRuntime(),
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    try:
        with pytest.raises(ValueError, match="must not carry operator receipts"):
            env.reset(options={"operator_ready_receipt": {"event": "ready"}})
    finally:
        env.close()


def test_yam_agent_env_per_step_robot_state_uses_measured_state_not_accepted_target(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    measured = np.zeros(14, dtype=np.float64)
    measured[[6, 13]] = 0.5
    runtime = FakeRuntime(measured, accepted_updates_measured_state=False)
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    action = measured.copy()
    action[0] = 0.4
    try:
        _observe_write_ready_reset(env, receipt_path)
        obs, _, _, _, info = env.chunk_step(action)
    finally:
        env.close()

    assert np.array_equal(obs["state"]["joint_position"], measured)
    assert np.array_equal(info["per_step"][0]["robot_state"]["qpos"], measured)
    assert np.array_equal(info["accepted_action"], action)


def test_yam_agent_env_return_all_frames_uses_standard_obs_list_contract(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    runtime = FakeRuntime(accepted_updates_measured_state=True)
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path, operator_receipt_path=receipt_path),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    actions = _valid_action_chunk(3)
    try:
        _observe_write_ready_reset(env, receipt_path)
        obs, _, _, _, info = env.chunk_step(actions, return_all_frames=True)
    finally:
        env.close()

    assert isinstance(obs, list)
    assert len(obs) == info["executed_actions"] == 3
    assert all("frames" in frame for frame in obs)


@pytest.mark.parametrize("terminal_kind", ["success", "limit"])
def test_yam_agent_env_holds_when_success_or_step_limit_stops_chunk(
    terminal_kind: str,
    tmp_path,
) -> None:
    receipt_path = tmp_path / "operator-receipt.json"
    env_holder: dict[str, YamAgentEnv] = {}

    def mark_success() -> None:
        if terminal_kind == "success":
            _write_operator_receipt(env_holder["env"], receipt_path, event="success")

    runtime = FakeRuntime(accepted_updates_measured_state=True, on_command=mark_success)
    step_lim = 10 if terminal_kind == "success" else 1
    env = YamAgentEnv(
        _yam_agent_config(
            tmp_path=tmp_path,
            step_lim=step_lim,
            operator_receipt_path=receipt_path,
        ),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    env_holder["env"] = env
    try:
        _observe_write_ready_reset(env, receipt_path)
        startup_holds = runtime.hold_calls
        _, _, terminated, truncated, _ = env.chunk_step(_valid_action_chunk(2))
    finally:
        env.close()

    assert runtime.hold_calls > startup_holds

    assert terminated is (terminal_kind == "success")
    assert truncated is (terminal_kind == "limit")


def test_yam_primitive_builds_pi05_three_view_batch_order() -> None:
    env = FakeToolkitEnv()
    model = FakeToolkitModel()
    primitive = YamPrimitives(env=env, model=model, check_cancelled=lambda: None)

    result = primitive.pi05_act(chunks=1)

    assert result["executed_steps"] == MODEL_SPEC.use_length
    obs = model.calls[0]
    assert obs["main_images"].shape == (1, 3, 4, 3)
    assert obs["extra_view_images"].shape == (1, 2, 3, 4, 3)
    assert np.array_equal(obs["main_images"][0], env.last_obs["frames"]["top"])
    assert np.array_equal(
        obs["extra_view_images"][0, 0], env.last_obs["frames"]["left"]
    )
    assert np.array_equal(
        obs["extra_view_images"][0, 1], env.last_obs["frames"]["right"]
    )
    assert obs["wrist_images"] is None
    assert obs["states"].shape == (1, 14)
    assert obs["task_descriptions"] == ["place the cube"]


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_yam_vla_localhost_rpc_round_trip_meta_and_actions(transport: str) -> None:
    model = FakeYamModel()
    facade = YamVLAFacade(model=model)

    with _served_facade(facade, transport) as rpc:
        client = BaseVLAClient(rpc)
        actions = client.predict(_valid_vla_observation(), options={"mode": "eval"})

    assert actions.shape == (1, MODEL_SPEC.use_length, 14)
    assert np.all(actions[0, :, 6] == pytest.approx(0.25))
    assert np.all(actions[0, :, 13] == pytest.approx(0.75))
    assert len(model.calls) == 1
    env_obs, mode = model.calls[0]
    assert mode == "eval"
    assert env_obs["main_images"].shape == (1, 3, 4, 3)
    assert env_obs["extra_view_images"].shape == (1, 2, 3, 4, 3)
    assert np.all(env_obs["extra_view_images"][0, 0] == 20)
    assert np.all(env_obs["extra_view_images"][0, 1] == 30)
    assert env_obs["wrist_images"] is None
    assert env_obs["states"].shape == (1, 14)


def test_yam_vla_contract_and_build_cfg_use_horizon30_use5_qpos14() -> None:
    cfg = build_model_cfg("/tmp/fake-yam-model", "/tmp/fake-norm-stats")

    assert MODEL_SPEC.camera_order == ("top", "left", "right")
    assert MODEL_SPEC.action_layout == "qpos14"
    assert MODEL_SPEC.action_horizon == 30
    assert MODEL_SPEC.use_length == 5
    assert cfg.model_path == "/tmp/fake-yam-model"
    assert cfg.precision == "bf16"
    assert cfg.num_action_chunks == MODEL_SPEC.action_horizon
    assert cfg.action_dim == 14
    assert cfg.openpi.task == "eval"
    assert cfg.openpi.model_action_dim == 32
    assert cfg.openpi.discrete_state_input is True
    assert "num_images_in_input" not in cfg.openpi
    assert "action_horizon" not in cfg.openpi
    assert "action_chunk" not in cfg.openpi
    assert "action_env_dim" not in cfg.openpi


@pytest.mark.parametrize(
    "bad_obs",
    [
        {
            **_valid_vla_observation(),
            "main_images": np.zeros((3, 4, 3), dtype=np.uint8),
        },
        {
            **_valid_vla_observation(),
            "extra_view_images": np.zeros((1, 1, 3, 4, 3), dtype=np.uint8),
        },
        {
            **_valid_vla_observation(),
            "states": np.full((1, 14), np.nan, dtype=np.float32),
        },
        {
            **_valid_vla_observation(),
            "task_descriptions": [""],
        },
    ],
)
def test_yam_vla_facade_rejects_bad_observations_before_model_call(
    bad_obs: dict[str, Any],
) -> None:
    model = FakeYamModel()
    facade = YamVLAFacade(model=model)

    with pytest.raises((TypeError, ValueError)):
        facade.predict(bad_obs)

    assert model.calls == []


@pytest.mark.parametrize(
    "bad_actions",
    [
        np.zeros((1, MODEL_SPEC.use_length, 13), dtype=np.float32),
        np.full((1, MODEL_SPEC.use_length, 14), np.nan, dtype=np.float32),
        np.repeat(
            np.array([[[0, 0, 0, 0, 0, 0, -0.1, 0, 0, 0, 0, 0, 0, 0.5]]]),
            MODEL_SPEC.use_length,
            axis=1,
        ),
        np.repeat(
            np.array([[[0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 1.1]]]),
            MODEL_SPEC.use_length,
            axis=1,
        ),
    ],
)
def test_yam_vla_facade_rejects_bad_policy_outputs(bad_actions: np.ndarray) -> None:
    model = FakeYamModel(actions=bad_actions)
    facade = YamVLAFacade(model=model)

    with pytest.raises(ValueError):
        facade.predict(_valid_vla_observation())


def test_yam_vla_rpc_returns_errors_for_bad_observation() -> None:
    facade = YamVLAFacade(model=FakeYamModel())
    bad_obs = {
        **_valid_vla_observation(),
        "extra_view_images": np.zeros((1, 2, 3, 4, 1), dtype=np.uint8),
    }

    with _served_facade(facade, "http") as rpc:
        with pytest.raises(RpcError, match="extra_view_images"):
            rpc.call("vla.predict", args=(bad_obs, {"mode": "eval"}), timeout_s=2.0)


def _make_yam_toolkit(
    tmp_path: Path,
    *,
    success_after_chunk: bool = False,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    model: FakeToolkitModel | None | object = _DEFAULT_TOOLKIT_MODEL,
) -> tuple[Any, FakeToolkitEnv, FakeToolkitModel | None, MemoryManager]:
    env = FakeToolkitEnv(success_after_chunk=success_after_chunk)
    if model is _DEFAULT_TOOLKIT_MODEL:
        model = FakeToolkitModel()
    assert model is None or isinstance(model, FakeToolkitModel)
    memory = MemoryManager(
        tmp_path / "memory",
        memory_access="inbox_write" if mode == "exploration" else "read_only",
        inbox_cell_tag="yam_place_cube_s12" if mode == "exploration" else None,
    )
    toolkit = YamToolkit(
        primitives_kwargs={"env": env, "model": model},
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=tmp_path / "state",
        run_output_dir=tmp_path / "run",
    )
    return toolkit, env, model, memory


def test_yam_toolkit_without_model_hides_pi05_but_keeps_primitives_and_reset(
    tmp_path,
) -> None:
    toolkit, _, _, _ = _make_yam_toolkit(
        tmp_path,
        mode="exploration",
        model=None,
    )
    try:
        tool_names = set(toolkit._tools)
    finally:
        toolkit.close()

    assert "pi05_act" not in tool_names
    assert {"move_to", "rotate_wrist", "set_gripper", "release", "reset"}.issubset(
        tool_names
    )


def test_yam_toolkit_finish_claim_cannot_create_verified_success(tmp_path) -> None:
    toolkit, env, _, _ = _make_yam_toolkit(tmp_path, success_after_chunk=False)
    try:
        result = toolkit.execute_tool(
            "finish",
            {"status": "success", "summary": "agent claimed success"},
        )
    finally:
        toolkit.close()

    assert env.eval_success is False
    assert result.is_finish is True
    assert result.result["verified_success"] is False
    assert result.result["status"] == "failure"


@pytest.mark.parametrize(
    ("arm", "target", "other_index"),
    [
        ("left", 0.2, 13),
        ("right", 0.8, 6),
    ],
)
def test_yam_primitives_set_gripper_reports_actual_value_and_preserves_other_arm(
    arm: str,
    target: float,
    other_index: int,
) -> None:
    env = FakeToolkitEnv()
    primitive = YamPrimitives(env=env, model=None, check_cancelled=lambda: None)
    before_other = float(env.last_obs["state"]["joint_position"][other_index])

    result = primitive.set_gripper(arm=arm, val=target, steps=2)

    gripper_index = 6 if arm == "left" else 13
    qpos = env.last_obs["state"]["joint_position"]
    assert result["success"] is True
    assert result["gripper_val"] == pytest.approx(target)
    assert qpos[gripper_index] == pytest.approx(target)
    assert qpos[other_index] == pytest.approx(before_other)


def test_yam_toolkit_pi05_act_rejects_model_side_episode_reset(tmp_path) -> None:
    env = FakeToolkitEnv(require_expected_episode_id=True)
    model = FakeToolkitModel(on_predict=env.force_new_episode_id)
    memory = MemoryManager(tmp_path / "memory", memory_access="read_only")
    toolkit = YamToolkit(
        primitives_kwargs={"env": env, "model": model},
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        mode="evaluation",
        state_output_dir=tmp_path / "state",
        run_output_dir=tmp_path / "run",
    )
    try:
        result = toolkit.execute_tool("pi05_act", {"chunks": 1})
    finally:
        toolkit.close()

    assert len(model.calls) == 1
    assert env.take_action_cnt == 0
    assert env.hold_calls == 1
    assert result.result["error"] == "stale episode_id refused before command"


def test_yam_toolkit_no_vla_primitives_recipe_and_memory_merge(tmp_path) -> None:
    toolkit, env, _, memory = _make_yam_toolkit(
        tmp_path,
        success_after_chunk=False,
        mode="exploration",
        attempts_per_session=1,
        model=None,
    )
    recipe_tag = "yam_place_cube_s12"
    technique_path = (
        tmp_path / "memory" / "_internal" / "inbox" / recipe_tag / "suite_technique.md"
    )
    technique = """---
scope: suite
suite: yam
regime: real
task_id: place_cube
task_language: place the cube
confidence: single-shot
evidence:
  cells:
    - yam_place_cube_s12
---
Use the left gripper to stabilize the object before opening the right gripper.
"""
    try:
        set_result = toolkit.execute_tool(
            "set_gripper", {"arm": "left", "val": 0.25, "steps": 2}
        )
        env.success_after_chunk = True
        release_result = toolkit.execute_tool(
            "release", {"arm": "right", "val": 1.0, "steps": 2}
        )
        write_result = toolkit.execute_tool(
            "write_text_file",
            {"path": str(technique_path), "content": technique},
        )
        recipe_path = Path(toolkit.write_recipe(recipe_tag))
    finally:
        toolkit.close()

    assert "pi05_act" not in set(toolkit._tools)
    assert set_result.result["state"]["episode_status"]["eval_success"] is False
    assert release_result.result["state"]["episode_status"]["eval_success"] is True
    assert set_result.result["log"]["result"]["gripper_val"] == pytest.approx(0.25)
    assert release_result.result["log"]["result"]["gripper_val"] == pytest.approx(1.0)
    assert write_result.result["bytes_written"] == len(technique.encode("utf-8"))
    assert env.take_action_cnt == 4
    assert recipe_path.exists()
    recipe_text = recipe_path.read_text(encoding="utf-8")
    assert recipe_text.count('"action": "set_gripper"') == 1
    assert recipe_text.count('"action": "release"') == 1

    merge = memory.merge_memory(
        cell_tag=recipe_tag,
        run_state_dir=tmp_path / "run",
        solved=True,
    )

    assert merge["suite"] == 1
    assert merge["task"] == 1
    assert (tmp_path / "memory" / "MEMORY.md").exists()
    assert (tmp_path / "memory" / "suite" / "suite_yam_real_tplace_cube.md").exists()
    assert (tmp_path / "memory" / "task_only" / f"{recipe_tag}_recipe.jsonl").exists()
    assert (tmp_path / "memory" / "task_only" / f"{recipe_tag}.json").exists()

    # A separate episode consumes the published corpus through normal tool bindings.
    next_toolkit, _, _, next_memory = _make_yam_toolkit(tmp_path, model=None)
    try:
        recalled = next_toolkit.execute_tool(
            "read_text_file",
            {"path": str(next_memory.root / "suite" / "suite_yam_real_tplace_cube.md")},
        )
        recipe = next_toolkit.execute_tool(
            "read_text_file",
            {
                "path": str(
                    next_memory.root / "task_only" / f"{recipe_tag}_recipe.jsonl"
                )
            },
        )
        rejected = next_toolkit.execute_tool(
            "write_text_file", {"path": str(technique_path), "content": "overwrite"}
        )
    finally:
        next_toolkit.close()
    assert "stabilize the object" in recalled.result["content"]
    assert '"action": "release"' in recipe.result["content"]
    assert "error" in rejected.result


def test_yam_toolkit_recipe_uses_only_current_episode_after_external_reset(
    tmp_path,
) -> None:
    toolkit, env, _, _ = _make_yam_toolkit(
        tmp_path,
        success_after_chunk=False,
        mode="exploration",
        model=None,
    )
    recipe_tag = "yam_place_cube_s12"
    try:
        old_result = toolkit.execute_tool(
            "set_gripper", {"arm": "left", "val": 0.25, "steps": 1}
        )
        old_episode = old_result.result["state"]["episode_status"]["episode_id"]
        env.force_new_episode_id()
        env.success_after_chunk = True
        new_result = toolkit.execute_tool(
            "release", {"arm": "right", "val": 1.0, "steps": 1}
        )
        new_episode = new_result.result["state"]["episode_status"]["episode_id"]
        recipe_path = Path(toolkit.write_recipe(recipe_tag))
    finally:
        toolkit.close()

    assert old_episode != new_episode
    assert recipe_path.exists()
    recipe_text = recipe_path.read_text(encoding="utf-8")
    assert '"action": "set_gripper"' not in recipe_text
    assert recipe_text.count('"action": "release"') == 1
    audit = json.loads((tmp_path / "run" / f"{recipe_tag}.json").read_text())
    assert audit["episode_status"]["episode_id"] == new_episode
    assert audit["recipe_actions"] == 1


def test_yam_toolkit_success_writes_recipe_and_memory_task_artifacts(tmp_path) -> None:
    toolkit, env, model, memory = _make_yam_toolkit(
        tmp_path,
        success_after_chunk=True,
        mode="exploration",
        attempts_per_session=2,
    )
    recipe_tag = "yam_place_cube_s12"
    try:
        assert toolkit.write_recipe(recipe_tag) == ""
        action_result = toolkit.execute_tool("pi05_act", {"chunks": 1})
        assert action_result.result["state"]["episode_status"]["eval_success"] is True
        assert toolkit.solved() is True
        recipe_path = Path(toolkit.write_recipe(recipe_tag))
    finally:
        toolkit.close()

    assert len(model.calls) == 1
    assert env.take_action_cnt == MODEL_SPEC.use_length
    assert recipe_path.exists()
    assert recipe_path.read_text(encoding="utf-8").count('"action": "pi05_act"') == 1
    audit_path = tmp_path / "run" / f"{recipe_tag}.json"
    assert audit_path.exists()

    merge = memory.merge_memory(
        cell_tag=recipe_tag,
        run_state_dir=tmp_path / "run",
        solved=True,
    )

    assert merge["task"] == 1
    assert (tmp_path / "memory" / "task_only" / f"{recipe_tag}_recipe.jsonl").exists()
    assert (tmp_path / "memory" / "task_only" / f"{recipe_tag}.json").exists()


def test_yam_toolkit_failure_does_not_write_recipe_or_merge_task_artifacts(
    tmp_path,
) -> None:
    toolkit, env, _, memory = _make_yam_toolkit(
        tmp_path,
        success_after_chunk=False,
        mode="exploration",
        attempts_per_session=1,
    )
    recipe_tag = "yam_place_cube_s12"
    try:
        action_result = toolkit.execute_tool("pi05_act", {"chunks": 1})
        assert action_result.result["state"]["episode_status"]["eval_success"] is False
        assert toolkit.solved() is False
        assert toolkit.write_recipe(recipe_tag) == ""
    finally:
        toolkit.close()

    merge = memory.merge_memory(
        cell_tag=recipe_tag,
        run_state_dir=tmp_path / "run",
        solved=False,
    )

    assert env.take_action_cnt == MODEL_SPEC.use_length
    assert merge["task"] == 0
    assert not (
        tmp_path / "memory" / "task_only" / f"{recipe_tag}_recipe.jsonl"
    ).exists()
    assert not (tmp_path / "memory" / "task_only" / f"{recipe_tag}.json").exists()


def test_env_step_rpc_rejects_multiple_actions_before_command(tmp_path) -> None:
    runtime = FakeRuntime()
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path),
        runtime=runtime,
        cameras=FakeCameraRig(),
        kinematics={"left": FakeKinematics(), "right": FakeKinematics()},
    )
    facade = YamEnvFacade(env)
    try:
        with pytest.raises(ValueError, match="single qpos14"):
            facade._dispatch("env.step", (_valid_action_chunk(2),), {})
        assert runtime.commands == []
    finally:
        facade.close()


@pytest.mark.parametrize("arm,offset", [("left", 0), ("right", 7)])
def test_single_arm_preserves_other_command_despite_measured_drift(arm, offset):
    env = FakeToolkitEnv()
    commanded = env.qpos.copy()
    commanded[:6] = 0.1
    commanded[7:13] = 0.2
    commanded[[6, 13]] = [0.3, 0.8]
    env.last_info["commanded_qpos"] = commanded.copy()
    primitives = YamPrimitives(env=env, check_cancelled=lambda: None)
    primitives.apply_qpos_updates([{"arm": arm, "arm_qpos": [0.4] * 6}])
    sent = env.qpos
    expected = commanded.copy()
    expected[offset : offset + 6] = 0.4
    np.testing.assert_array_equal(sent, expected)


def test_camera_meta_and_render_remain_bound_to_observed_pose(tmp_path):
    runtime = FakeRuntime()
    env = YamAgentEnv(
        _yam_agent_config(tmp_path=tmp_path),
        runtime=runtime,
        cameras=FreshSnapshotCameraRig(),
        kinematics={side: FakeKinematics() for side in ("left", "right")},
    )
    try:
        observation, info = env.observe()
        runtime.qpos[0] += 0.2
        runtime.qpos[7] += 0.3
        for side in ("left", "right"):
            meta = env.get_camera_meta(side)
            np.testing.assert_array_equal(
                meta["cam2world_cv"],
                observation["views"][side]["camera_meta"]["cam2world_cv"],
            )
            np.testing.assert_array_equal(
                env.render_camera(side), observation["views"][side]["rgb"]
            )
        _, next_info = env.observe()
        np.testing.assert_array_equal(
            next_info["commanded_qpos"], info["commanded_qpos"]
        )
    finally:
        env.close()


def test_can_lease_conflict_and_release(tmp_path):
    first = HardwareLease(["can_left", "can_right"], lock_dir=tmp_path)
    second = HardwareLease(["can_right"], lock_dir=tmp_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="owned"):
            second.acquire()
    finally:
        first.close()
    second.acquire()
    second.close()


def test_existing_socketcan_subscriptions_are_rejected(tmp_path):
    lease = HardwareLease(["can_left", "can_right"], lock_dir=tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(RuntimeError, match="unavailable"):
        lease.check_subscriptions(proc_root=proc)
    entries = proc / "rcvlist_all"
    entries.write_text(
        "receive list 'rx_all':\n  (any: no entry)\n  (can_left: no entry)\n"
    )
    lease.check_subscriptions(proc_root=proc)
    entries.write_text(" device can_id can_mask\n can_left 000 00000000\n")
    with pytest.raises(RuntimeError, match="Existing CAN"):
        lease.check_subscriptions(proc_root=proc)


def test_failed_ownership_cannot_be_bypassed_by_second_observe(tmp_path, monkeypatch):
    rig = FreshSnapshotCameraRig()
    env = YamAgentEnv(_yam_agent_config(tmp_path=tmp_path), cameras=rig)
    monkeypatch.setattr(HardwareLease, "acquire", lambda self: None)
    monkeypatch.setattr(HardwareLease, "close", lambda self: None)

    def busy(self):
        raise RuntimeError("Existing CAN subscription")

    monkeypatch.setattr(HardwareLease, "check_subscriptions", busy)
    with pytest.raises(RuntimeError, match="Existing CAN"):
        env.observe()
    with pytest.raises(RuntimeError, match="startup failed"):
        env.observe()
    assert rig.open_calls == 0
    assert env._hardware_lease is None
    env.close()


def test_close_can_retry_after_runtime_cleanup_failure(tmp_path, monkeypatch):
    runtime = FakeRuntime()
    env = YamAgentEnv(_yam_agent_config(tmp_path=tmp_path), runtime=runtime)
    calls = []

    def close():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("disconnect failure")

    monkeypatch.setattr(runtime, "close", close)
    with pytest.raises(RuntimeError, match="fully close"):
        env.close()
    assert not env._closed
    env.close()
    assert env._closed
    assert len(calls) == 2


def test_measured_transition_uses_feedback_and_rejects_tracking_lag(
    tmp_path, monkeypatch
):
    runtime = FakeRuntime()
    config = _yam_agent_config(tmp_path=tmp_path)
    config["max_tracking_error_rad"] = 0.1
    env = YamAgentEnv(config, runtime=runtime)
    measured = runtime.qpos.copy()
    measured[0] = 0.05
    runtime.qpos = measured.copy()
    target = measured.copy()
    target[0] = 0.08
    checked = []

    def guard(start, finish):
        checked.append((start.copy(), finish.copy()))
        return {"ok": True}

    monkeypatch.setattr(env.geometry, "check_qpos_transition", guard)
    env._check_measured_transition(target)
    np.testing.assert_array_equal(checked[0][0], measured)
    np.testing.assert_array_equal(checked[0][1], target)
    target[0] = 0.4
    with pytest.raises(RuntimeError, match="tracking error"):
        env._check_measured_transition(target)
    assert runtime.commands == []
    env.close()


@pytest.mark.parametrize("table_z", [None, float("nan"), float("inf")])
def test_site_requires_verified_table_before_any_motion(tmp_path, table_z):
    config = _yam_agent_config(tmp_path=tmp_path)
    config["require_table_guard"] = True
    config["table_z"] = table_z
    runtime = FakeRuntime()
    if table_z is not None and not np.isfinite(table_z):
        with pytest.raises(ValueError, match="table_z must be finite"):
            YamAgentEnv(config, runtime=runtime)
        assert runtime.commands == []
        return
    env = YamAgentEnv(config, runtime=runtime)
    env._operator_ready_receipt = {"event": "ready"}
    with pytest.raises(RuntimeError, match="verified table geometry"):
        env._require_ready_for_motion()
    assert runtime.commands == []
    env.close()
