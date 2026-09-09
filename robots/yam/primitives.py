# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""YAM primitives built on the env and Pi0.5 RPC contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.yam.contracts import MODEL_SPEC, validate_actions
from robots.yam.env_client import YamEnvClient
from rpent.robots.components.vla_client_base import BaseVLAClient


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(abs(np.dot(left, right)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


class YamPrimitives:
    def __init__(
        self,
        *,
        env: YamEnvClient,
        model: BaseVLAClient | None = None,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self.model = model
        self._check_cancelled = check_cancelled
        self._frames: list[np.ndarray] = []

    def _record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._frames = []
        return frames

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        budget_exhausted = int(status.get("take_action_cnt", 0)) >= int(
            status.get("step_lim", 0)
        )
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "operator_or_env_success"
        elif budget_exhausted:
            stop_reason = "budget_exhausted"
        elif completed:
            stop_reason = "completed"
        else:
            stop_reason = "runtime_failure"
        return {
            "completed": completed,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": stop_reason,
        }

    def reset(self) -> dict[str, Any]:
        _, info = self.env.reset()
        return {**info, "success": True}

    def _build_policy_observation(self, *, prompt: str | None = None) -> dict[str, Any]:
        self.env.observe()
        frames = self.env.last_obs["frames"]
        instruction = self.env.get_task_language()
        policy_instruction = (
            prompt if prompt is not None and prompt.strip() else instruction
        )
        return {
            "main_images": np.asarray(frames["top"])[None],
            "wrist_images": None,
            "extra_view_images": np.stack([frames["left"], frames["right"]])[None],
            "states": np.asarray(
                self.env.last_obs["state"]["joint_position"], dtype=np.float32
            )[None],
            "task_descriptions": [policy_instruction],
        }

    def _record_chunk_payload(self, payload: Any) -> None:
        observations: list[Any]
        if isinstance(payload, list):
            observations = payload
        elif isinstance(payload, dict):
            observations = [payload]
        else:
            observations = []
        for obs in observations:
            if isinstance(obs, dict):
                frame = obs.get("frames", {}).get("top")
                if frame is not None:
                    self._record_frame(frame)

    def pi05_act(
        self,
        *,
        chunks: int = 1,
        use_length: int = MODEL_SPEC.use_length,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "YAM VLA is not connected; pi05_act requires a trained --vla-endpoint"
            )
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if int(use_length) != MODEL_SPEC.use_length:
            raise ValueError(f"YAM Pi0.5 requires use_length={MODEL_SPEC.use_length}")
        executed = 0
        requested = int(chunks) * MODEL_SPEC.use_length
        native_prompt = None
        for _ in range(int(chunks)):
            self._check_cancelled()
            status = self.env.last_info["episode_status"]
            if status.get("eval_success") is True or int(
                status["take_action_cnt"]
            ) >= int(status["step_lim"]):
                break
            native_prompt = (
                prompt
                if prompt is not None and prompt.strip()
                else self.env.get_task_language()
            )
            observation = self._build_policy_observation(prompt=prompt)
            episode_id = self.env.last_info["episode_status"]["episode_id"]
            actions = validate_actions(np.asarray(self.model.predict(observation))[0])[
                : MODEL_SPEC.use_length
            ]
            payload, _, _, _, info = self.env.chunk_step(
                actions,
                action_type="qpos",
                expected_episode_id=episode_id,
                return_all_frames=self.env.execution_capabilities.get(
                    "chunk_step_all_frames"
                )
                is True,
            )
            self._record_chunk_payload(payload)
            count = int(info.get("executed_actions", 0))
            executed += count
        status = self.env.last_info["episode_status"]
        return {
            **self._completion(requested=requested, executed=executed, status=status),
            "success": executed == requested,
            "prompt": native_prompt,
            "episode_status": status,
        }

    @staticmethod
    def _validate_qpos_updates_request(updates: Any) -> list[dict[str, Any]]:
        if not isinstance(updates, list) or not updates:
            raise ValueError("qpos updates must contain at least one update")
        normalized = []
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("qpos update must be a mapping")
            arm = update.get("arm")
            if arm not in ("left", "right"):
                raise ValueError("arm must be 'left' or 'right'")
            if update.get("arm_qpos") is None and update.get("gripper") is None:
                raise ValueError("qpos update must set arm_qpos and/or gripper")
            item: dict[str, Any] = {"arm": arm}
            if update.get("arm_qpos") is not None:
                arm_qpos = np.asarray(update["arm_qpos"], dtype=np.float64)
                if arm_qpos.shape != (6,) or not np.isfinite(arm_qpos).all():
                    raise ValueError("arm_qpos must be finite and have shape (6,)")
                item["arm_qpos"] = arm_qpos
            if update.get("gripper") is not None:
                gripper = float(update["gripper"])
                if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
                    raise ValueError("gripper must be finite and within [0,1]")
                item["gripper"] = gripper
            normalized.append(item)
        return normalized

    def apply_qpos_updates(
        self, updates: list[dict[str, Any]], *, expected_episode_id: str | None = None
    ) -> dict[str, Any]:
        updates = self._validate_qpos_updates_request(updates)
        self._check_cancelled()
        state = np.asarray(
            self.env.last_obs["state"]["joint_position"], dtype=np.float64
        )
        action = np.asarray(
            self.env.last_info.get("commanded_qpos", state), dtype=np.float64
        ).copy()
        actions = []
        for update in updates:
            offset = 0 if update["arm"] == "left" else 7
            if "arm_qpos" in update:
                action[offset : offset + 6] = update["arm_qpos"]
            if "gripper" in update:
                action[offset + 6] = update["gripper"]
            actions.append(action.copy())
        actions_array = validate_actions(actions)
        payload, _, _, _, info = self.env.chunk_step(
            actions_array,
            action_type="qpos",
            expected_episode_id=expected_episode_id
            or self.env.last_info["episode_status"]["episode_id"],
            return_all_frames=self.env.execution_capabilities.get(
                "chunk_step_all_frames"
            )
            is True,
        )
        self._record_chunk_payload(payload)
        episode_status = info["episode_status"]
        executed = int(info.get("executed_actions", 0))
        return {
            "action_type": "qpos",
            "requested_actions": len(updates),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def move_to(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        robot_state = self.env.last_info["robot_state"]
        if quat is None:
            key = "left_eef_pose" if arm == "left" else "right_eef_pose"
            quat = np.asarray(robot_state[key], dtype=np.float64)[3:].tolist()
        target = np.asarray([*xyz, *quat], dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("target pose must be finite xyz + wxyz")
        episode_id = self.env.last_info["episode_status"]["episode_id"]
        planned = self.env.plan_arm_path(arm, target)
        if planned.get("status") != "Success" or planned.get("position") is None:
            return {
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "plan_status": planned.get("status"),
                "hint": planned.get("reason", "target may be unreachable or unsafe"),
            }
        path = np.asarray(planned["position"], dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError(
                f"YAM plan_arm_path returned invalid path shape {path.shape}"
            )
        updates = [
            {"arm": arm, "arm_qpos": waypoint, "gripper": gripper} for waypoint in path
        ]
        execution = self.apply_qpos_updates(updates, expected_episode_id=episode_id)
        executed = int(execution.get("executed_actions", 0))
        status = execution["episode_status"]
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        measured_pose = np.asarray(
            self.env.last_info["robot_state"][key], dtype=np.float64
        )
        position_error_m = float(np.linalg.norm(measured_pose[:3] - target[:3]))
        rotation_error_rad = _quat_angle_rad(measured_pose[3:], target[3:])
        position_tolerance_m = 0.025
        rotation_tolerance_rad = 0.25
        reached_target = (
            executed == len(updates)
            and position_error_m <= position_tolerance_m
            and rotation_error_rad <= rotation_tolerance_rad
        )
        completion = self._completion(
            requested=len(updates), executed=executed, status=status
        )
        if not reached_target and completion["stop_reason"] == "completed":
            completion["stop_reason"] = "target_not_reached"
        return {
            **execution,
            **completion,
            "success": reached_target,
            "plan_status": planned["status"],
            "waypoints": len(path),
            "measured_pose": measured_pose.tolist(),
            "target_pose": target.tolist(),
            "position_error_m": position_error_m,
            "rotation_error_rad": rotation_error_rad,
            "position_tolerance_m": position_tolerance_m,
            "rotation_tolerance_rad": rotation_tolerance_rad,
        }

    def rotate_wrist(
        self,
        *,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        state = self.env.last_info["robot_state"]
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        pose = np.asarray(state[key], dtype=np.float64)
        yaw = np.deg2rad(float(delta_yaw_deg))
        world_z = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=_qmult(world_z, pose[3:]).tolist(),
            gripper=gripper,
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def set_gripper(
        self,
        *,
        arm: str,
        val: float,
        steps: int = 10,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        state = np.asarray(
            self.env.last_obs["state"]["joint_position"], dtype=np.float64
        )
        current = float(state[6 if arm == "left" else 13])
        target = float(val)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("val must be finite and within [0,1]")
        values = [
            current + (target - current) * i / int(steps)
            for i in range(1, int(steps) + 1)
        ]
        execution = self.apply_qpos_updates(
            [{"arm": arm, "gripper": value} for value in values]
        )
        executed = int(execution.get("executed_actions", 0))
        now = np.asarray(self.env.last_obs["state"]["joint_position"], dtype=np.float64)
        return {
            **execution,
            **self._completion(
                requested=len(values),
                executed=executed,
                status=execution["episode_status"],
            ),
            "success": executed == len(values),
            "gripper_val": float(now[6 if arm == "left" else 13]),
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(arm=arm, val=val, steps=steps)
