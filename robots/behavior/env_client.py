"""BEHAVIOR env RPC client."""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.rpc_driver.base import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 1800.0,
    "env.chunk_step": 1800.0,
    "env.pi0_chunk_step": 1800.0,
    "env.pi0_navigate_to_chunk_step": 1800.0,
    "env.pi0_nav_pick_chunk_step": 1800.0,
    "env.save_robot_state_checkpoint": 120.0,
    "env.restore_robot_state_checkpoint": 1800.0,
    "env.finalize_paused_runtime": 120.0,
    "env.current_observation": 120.0,
    "env.inspect_post_pick_state": 120.0,
    "env.observe": 120.0,
    "env.declare_button_visibility": 120.0,
    "env.project_button": 120.0,
    "env.evaluate_prepress_geometry": 120.0,
    "env.prepress_move_to": 1800.0,
    "env.prepress_rotate_wrist": 1800.0,
    "env.save_prepress_checkpoint": 120.0,
    "env.save_post_pick_debug_mirror": 1800.0,
    "env.pixel_to_world": 120.0,
    "env.navigate_to": 1800.0,
    "env.move_to": 1800.0,
    "env.pick": 1800.0,
    "env.rotate_wrist": 1800.0,
    "env.press": 1800.0,
    "env.release": 1800.0,
}


class BehaviorEnvClient:
    """Remote implementation of the BEHAVIOR single-env protocol."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict[str, Any],
    ) -> None:
        self._client = client
        self.episode_done = False
        self.total_env_steps = 0
        self.vla_endpoint: str | None = None
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=_TIMEOUT_S["default"],
        )
        if not isinstance(server_meta, dict):
            raise RuntimeError(
                f"env_meta must be a mapping, got {type(server_meta)!r}"
            )
        mismatches = {
            key: {"expected": expected, "actual": server_meta.get(key)}
            for key, expected in expected_meta.items()
            if server_meta.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"env_meta mismatch: {mismatches!r}"
            )
        self.server_meta = dict(server_meta)

    def reset(self) -> tuple[dict[str, Any], Any]:
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.episode_done = False
        self.total_env_steps = 0
        return ret

    def _track_step_result(self, ret: tuple[Any, Any, Any, Any, Any]) -> None:
        _, _, term, trunc, info = ret
        if np.asarray(term).any() or np.asarray(trunc).any():
            self.episode_done = True
        if isinstance(info, dict):
            if bool((info.get("done") or {}).get("success")):
                self.episode_done = True
            rpent = info.get("_rpent")
            if isinstance(rpent, dict) and "total_env_steps" in rpent:
                self.total_env_steps = int(rpent["total_env_steps"])

    def chunk_step(self, actions) -> tuple[Any, Any, Any, Any, Any]:
        assert not self.episode_done, "env.chunk_step called after episode done"
        # The local agent and remote BEHAVIOR runtime may use incompatible
        # NumPy major versions. Keep ndarray pickle internals off this boundary.
        wire_actions = np.asarray(actions, dtype=np.float32).tolist()
        ret = self._client.call(
            "env.chunk_step",
            args=(wire_actions,),
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        self._track_step_result(ret)
        return ret

    def pi0_chunk_step(
        self,
        actions,
        *,
        hand: str,
        gripper_closed_threshold: float = 0.045,
        required_closed_steps: int = 3,
        stop_on_candidate: bool = False,
    ) -> tuple[Any, Any, Any, Any, Any]:
        assert not self.episode_done, "env.pi0_chunk_step called after episode done"
        wire_actions = np.asarray(actions, dtype=np.float32).tolist()
        ret = self._client.call(
            "env.pi0_chunk_step",
            args=(wire_actions,),
            kwargs={
                "hand": hand,
                "gripper_closed_threshold": gripper_closed_threshold,
                "required_closed_steps": required_closed_steps,
                "stop_on_candidate": stop_on_candidate,
            },
            timeout_s=_TIMEOUT_S["env.pi0_chunk_step"],
        )
        self._track_step_result(ret)
        return ret

    def pi0_navigate_to_chunk_step(
        self,
        actions,
        *,
        segment_index: int,
        chunk_index: int,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Execute one bounded, base-only Pi0 visual-navigation chunk."""
        assert not self.episode_done, (
            "env.pi0_navigate_to_chunk_step called after episode done"
        )
        wire_actions = np.asarray(actions, dtype=np.float32).tolist()
        ret = self._client.call(
            "env.pi0_navigate_to_chunk_step",
            args=(wire_actions,),
            kwargs={
                "segment_index": int(segment_index),
                "chunk_index": int(chunk_index),
            },
            timeout_s=_TIMEOUT_S["env.pi0_navigate_to_chunk_step"],
        )
        self._track_step_result(ret)
        return ret

    def pi0_nav_pick_chunk_step(
        self,
        actions,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, Any, Any, Any]:
        assert not self.episode_done, (
            "env.pi0_nav_pick_chunk_step called after episode done"
        )
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (32, 23):
            raise ValueError(
                f"pi0_nav_pick requires one complete [32,23] chunk, got {action_array.shape}"
            )
        ret = self._client.call(
            "env.pi0_nav_pick_chunk_step",
            args=(action_array.tolist(),),
            kwargs={"chunk_index": int(chunk_index)},
            timeout_s=_TIMEOUT_S["env.pi0_nav_pick_chunk_step"],
        )
        self._track_step_result(ret)
        return ret

    def save_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str,
        stage: str,
        held_hand: str,
        press_hand: str,
        object_name: str,
        require_current_grasp: bool = True,
        visual_review: bool = True,
    ) -> dict[str, Any]:
        return self._planner_call(
            "save_robot_state_checkpoint",
            checkpoint_name=checkpoint_name,
            stage=stage,
            held_hand=held_hand,
            press_hand=press_hand,
            object_name=object_name,
            require_current_grasp=require_current_grasp,
            visual_review=visual_review,
        )

    def restore_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str,
        checkpoint_path: str | None = None,
        mode: str = "plan_and_execute",
        keep_held_gripper_closed: bool = True,
        require_object_still_held: bool = True,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        return self._planner_call(
            "restore_robot_state_checkpoint",
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            mode=mode,
            keep_held_gripper_closed=keep_held_gripper_closed,
            require_object_still_held=require_object_still_held,
            timeout_s=timeout_s,
        )

    def finalize_paused_runtime(self, vla_status: dict[str, Any]) -> dict[str, Any]:
        payload = dict(vla_status)
        if self.vla_endpoint:
            payload.setdefault("endpoint", self.vla_endpoint)
        ret = self._client.call(
            "env.finalize_paused_runtime",
            args=(payload,),
            timeout_s=_TIMEOUT_S["env.finalize_paused_runtime"],
        )
        if not isinstance(ret, dict):
            raise RuntimeError("env.finalize_paused_runtime returned a non-dict result")
        return ret

    def current_observation(self) -> tuple[dict[str, Any], Any]:
        """Refresh synchronized sensors without resetting or advancing physics."""
        ret = self._client.call(
            "env.current_observation",
            timeout_s=_TIMEOUT_S["env.current_observation"],
        )
        if not isinstance(ret, tuple) or len(ret) != 2:
            raise RuntimeError("env.current_observation returned an invalid result")
        observation, info = ret
        if isinstance(info, dict):
            rpent = info.get("_rpent")
            if isinstance(rpent, dict) and "total_env_steps" in rpent:
                self.total_env_steps = int(rpent["total_env_steps"])
            if bool((info.get("done") or {}).get("success")):
                self.episode_done = True
        return observation, info

    def inspect_post_pick_state(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_1",
    ) -> dict[str, Any]:
        return self._planner_call(
            "inspect_post_pick_state",
            checkpoint_name=checkpoint_name,
        )

    def get_env_meta(self) -> dict[str, Any]:
        return self._client.call("env.get_env_meta", timeout_s=_TIMEOUT_S["default"])

    def _planner_call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        rpc_timeout_s = _TIMEOUT_S.get(
            f"env.{method}", _TIMEOUT_S["default"]
        )
        requested_timeout = kwargs.get("timeout_s")
        if requested_timeout is not None:
            # The primitive owns the hard deadline.  Keep a bounded transport
            # grace period for serializing its structured timeout result rather
            # than leaving every planner RPC blocked for the global 30 minutes.
            rpc_timeout_s = min(
                rpc_timeout_s,
                max(30.0, float(requested_timeout) + 60.0),
            )
        ret = self._client.call(
            f"env.{method}",
            kwargs=kwargs,
            timeout_s=rpc_timeout_s,
        )
        if not isinstance(ret, dict):
            raise RuntimeError(f"env.{method} returned non-dict result: {type(ret)!r}")
        if "total_env_steps" in ret:
            self.total_env_steps = int(ret["total_env_steps"])
        if bool(ret.get("task_success", False)):
            self.episode_done = True
        return ret

    def observe(self, *, camera: str) -> dict[str, Any]:
        return self._planner_call("observe", camera=camera)

    def declare_button_visibility(
        self,
        *,
        camera: str,
        frame_id: str,
        button_visible: bool,
        positive_signature: dict[str, bool] | None = None,
        negative_case: str | None = None,
        bbox_xyxy: list[float] | None = None,
        center_uv: list[float] | None = None,
    ) -> dict[str, Any]:
        return self._planner_call(
            "declare_button_visibility",
            camera=camera,
            frame_id=frame_id,
            button_visible=button_visible,
            positive_signature=positive_signature,
            negative_case=negative_case,
            bbox_xyxy=bbox_xyxy,
            center_uv=center_uv,
        )

    def project_button(
        self,
        *,
        gate_id: str,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        return self._planner_call(
            "project_button",
            gate_id=gate_id,
            depth_window_px=depth_window_px,
        )

    def evaluate_prepress_geometry(
        self,
        *,
        projection_id: str,
        max_line_distance_m: float = 0.010,
        max_opposition_angle_deg: float = 15.0,
        min_axial_standoff_m: float = 0.03,
        max_axial_standoff_m: float = 0.06,
    ) -> dict[str, Any]:
        return self._planner_call(
            "evaluate_prepress_geometry",
            projection_id=projection_id,
            max_line_distance_m=max_line_distance_m,
            max_opposition_angle_deg=max_opposition_angle_deg,
            min_axial_standoff_m=min_axial_standoff_m,
            max_axial_standoff_m=max_axial_standoff_m,
        )

    def prepress_move_to(
        self,
        *,
        role: str = "held",
        button_goal: dict[str, Any],
        plan_only: bool = False,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        return self._planner_call(
            "prepress_move_to",
            role=role,
            button_goal=button_goal,
            plan_only=plan_only,
            timeout_s=timeout_s,
        )

    def prepress_rotate_wrist(
        self,
        *,
        role: str = "held",
        target_quat_xyzw: list[float] | None = None,
        relative_axis_angle: list[float] | None = None,
        frame: str = "eef",
        plan_only: bool = False,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        return self._planner_call(
            "prepress_rotate_wrist",
            role=role,
            target_quat_xyzw=target_quat_xyzw,
            relative_axis_angle=relative_axis_angle,
            frame=frame,
            plan_only=plan_only,
            timeout_s=timeout_s,
        )

    def save_prepress_checkpoint(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_2",
        stage: str = "pre_press_alignment",
        visual_review: bool = True,
    ) -> dict[str, Any]:
        return self._planner_call(
            "save_prepress_checkpoint",
            checkpoint_name=checkpoint_name,
            stage=stage,
            visual_review=visual_review,
        )

    def save_post_pick_debug_mirror(self) -> dict[str, Any]:
        """Persist the internal debug-only scene mirror after finalized state1."""

        return self._planner_call("save_post_pick_debug_mirror")

    def pixel_to_world(
        self,
        *,
        camera: str,
        frame_id: str,
        u: int,
        v: int,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        return self._planner_call(
            "pixel_to_world",
            camera=camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
            output_frame=output_frame,
        )

    def navigate_to(
        self,
        *,
        hand: str,
        target_xyz: list[float],
        frame: str = "world",
        standoff_m: float = 0.85,
        timeout_s: float = 90,
    ) -> dict[str, Any]:
        return self._planner_call(
            "navigate_to",
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            standoff_m=standoff_m,
            timeout_s=timeout_s,
        )

    def move_to(
        self,
        *,
        hand: str,
        target_xyz: list[float],
        frame: str = "world",
        target_quat_xyzw: list[float] | None = None,
        plan_only: bool = False,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.087,
        timeout_s: float = 45,
    ) -> dict[str, Any]:
        return self._planner_call(
            "move_to",
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            target_quat_xyzw=target_quat_xyzw,
            plan_only=plan_only,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            timeout_s=timeout_s,
        )

    def pick(
        self,
        *,
        hand: str,
        target_xyz: list[float],
        approach_vector: list[float] | None = None,
        grasp_quat_xyzw: list[float] | None = None,
        pregrasp_offset_m: float = 0.08,
        lift_m: float = 0.08,
        timeout_s: float = 90,
    ) -> dict[str, Any]:
        return self._planner_call(
            "pick",
            hand=hand,
            target_xyz=target_xyz,
            approach_vector=approach_vector,
            grasp_quat_xyzw=grasp_quat_xyzw,
            pregrasp_offset_m=pregrasp_offset_m,
            lift_m=lift_m,
            timeout_s=timeout_s,
        )

    def rotate_wrist(
        self,
        *,
        hand: str,
        target_quat_xyzw: list[float] | None = None,
        relative_axis_angle: list[float] | None = None,
        frame: str = "world",
        timeout_s: float = 45,
    ) -> dict[str, Any]:
        if (target_quat_xyzw is None) == (relative_axis_angle is None):
            raise ValueError(
                "rotate_wrist requires exactly one of target_quat_xyzw or "
                "relative_axis_angle"
            )
        return self._planner_call(
            "rotate_wrist",
            hand=hand,
            target_quat_xyzw=target_quat_xyzw,
            relative_axis_angle=relative_axis_angle,
            frame=frame,
            timeout_s=timeout_s,
        )

    def press(
        self,
        *,
        hand: str,
        target_xyz: list[float],
        press_direction: list[float] | None = None,
        approach_distance_m: float = 0.04,
        press_depth_m: float = 0.012,
        timeout_s: float = 60,
    ) -> dict[str, Any]:
        return self._planner_call(
            "press",
            hand=hand,
            target_xyz=target_xyz,
            press_direction=press_direction,
            approach_distance_m=approach_distance_m,
            press_depth_m=press_depth_m,
            timeout_s=timeout_s,
        )

    def release(
        self,
        *,
        hand: str,
        opening: float = 1.0,
        retreat_vector: list[float] | None = None,
        retreat_m: float = 0.03,
        timeout_s: float = 30,
    ) -> dict[str, Any]:
        return self._planner_call(
            "release",
            hand=hand,
            opening=opening,
            retreat_vector=retreat_vector,
            retreat_m=retreat_m,
            timeout_s=timeout_s,
        )

    def close(self) -> None:
        """Close only the client transport; runtime ownership stays with provider."""
        self._client.close()
