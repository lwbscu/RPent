# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPC client for the YAM real-robot env server."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.yam.contracts import YAM_CAMERA_NAMES, YAM_STATUS_KEYS, validate_actions
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class YamEnvClient(BaseEnvClient):
    """Client for one operator-supervised YAM env endpoint.

    Unlike the common base constructor, this client does not reset on connect.
    Reset changes the operator-approved episode, so construction performs
    metadata validation through the common base and then takes an explicit
    ``env.observe`` snapshot. Observe may initialize hardware.
    """

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict[str, Any],
    ) -> None:
        super().__init__(
            client,
            expected_meta=expected_meta,
            reset_on_connect=False,
        )
        self.server_meta = dict(expected_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )
        self.observe()

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(f"{method} must return a {size}-item tuple, got {result!r}")
        return tuple(result)

    @staticmethod
    def _require_episode_status(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise TypeError(f"YAM info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        if not isinstance(status, dict):
            raise TypeError(f"YAM episode_status must be a mapping, got {status!r}")
        missing = [key for key in YAM_STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"YAM episode_status is missing {missing}: {status!r}")
        return status

    @staticmethod
    def _require_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError(f"YAM observation must be a mapping, got {observation!r}")
        frames = observation.get("frames")
        if not isinstance(frames, dict):
            raise TypeError("YAM observation.frames must be a mapping")
        for name in YAM_CAMERA_NAMES:
            if name not in frames:
                raise ValueError(f"YAM observation is missing {name!r} camera")
        state = observation.get("state")
        if not isinstance(state, dict):
            raise TypeError("YAM observation.state must be a mapping")
        qpos = np.asarray(state.get("joint_position"), dtype=np.float64)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError(
                "YAM observation.state.joint_position must be finite qpos14"
            )
        return observation

    def observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._client.call(
            "env.observe", timeout_s=self._TIMEOUT_S["env.render_camera"]
        )
        observation, info = self._require_result_tuple(result, 2, "env.observe")
        self.last_obs = self._require_observation(observation)
        self.last_info = info
        self._require_episode_status(info)
        return self.last_obs, info

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._client.call(
            "env.reset",
            timeout_s=self._TIMEOUT_S["env.reset"],
        )
        observation, info = self._require_result_tuple(result, 2, "env.reset")
        self.last_obs = self._require_observation(observation)
        self.last_info = info
        self._require_episode_status(info)
        return self.last_obs, info

    def step(
        self,
        action,
        *,
        action_type: str = "qpos",
        expected_episode_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        flat = validate_actions(action, action_type=action_type)
        if flat.shape[0] != 1:
            raise ValueError("YAM step requires one qpos14 action")
        result = self._client.call(
            "env.step",
            args=(flat[0],),
            kwargs={
                "action_type": action_type,
                "expected_episode_id": expected_episode_id
                or self.last_info["episode_status"]["episode_id"],
            },
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        result = self._require_result_tuple(result, 5, "env.step")
        obs, _, _, _, info = result
        self.last_obs = self._require_observation(obs)
        self.last_info = info
        self._require_episode_status(info)
        return result

    def chunk_step(
        self,
        actions,
        *,
        action_type: str = "qpos",
        return_all_frames: bool = False,
        expected_episode_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        flat = validate_actions(actions, action_type=action_type)
        result = self._client.call(
            "env.chunk_step",
            args=(flat,),
            kwargs={
                "action_type": action_type,
                "return_all_frames": return_all_frames,
                "expected_episode_id": expected_episode_id
                or self.last_info["episode_status"]["episode_id"],
            },
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        obs_field, _, _, _, info = result
        if isinstance(obs_field, list):
            if not obs_field:
                raise TypeError("YAM chunk_step returned no observations")
            self.last_obs = self._require_observation(obs_field[-1])
        elif isinstance(obs_field, dict):
            self.last_obs = self._require_observation(obs_field)
        else:
            raise TypeError(
                "YAM chunk_step must return an observation dict or a list of observations"
            )
        self.last_info = info
        self._require_episode_status(info)
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        if camera_name not in YAM_CAMERA_NAMES:
            raise ValueError(
                f"unknown YAM camera {camera_name!r}; available={list(YAM_CAMERA_NAMES)}"
            )
        return super().render_camera(camera_name, depth=depth)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in YAM_CAMERA_NAMES:
            raise ValueError(
                f"unknown YAM camera {camera_name!r}; available={list(YAM_CAMERA_NAMES)}"
            )
        return super().get_camera_meta(camera_name)

    def get_task_language(self) -> str:
        result = super().get_task_language()
        if not isinstance(result, str):
            raise TypeError(f"YAM task language must be a string: {result!r}")
        return result

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=120.0,
        )

    def request_stop(self) -> dict[str, Any]:
        return self._client.call(
            "env.request_stop",
            timeout_s=5.0,
        )
