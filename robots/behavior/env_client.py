"""BEHAVIOR env RPC client."""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.rpc_driver.base import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 1800.0,
    "env.chunk_step": 1800.0,
    "env.observe": 120.0,
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
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=_TIMEOUT_S["default"],
        )
        if server_meta != expected_meta:
            raise RuntimeError(
                f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}"
            )

    def reset(self) -> tuple[dict[str, Any], Any]:
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.episode_done = False
        return ret

    def chunk_step(self, actions) -> tuple[Any, Any, Any, Any, Any]:
        assert not self.episode_done, "env.chunk_step called after episode done"
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, info = ret
        if np.asarray(term).any() or np.asarray(trunc).any():
            self.episode_done = True
        if isinstance(info, dict) and bool((info.get("done") or {}).get("success")):
            self.episode_done = True
        return ret

    def get_env_meta(self) -> dict[str, Any]:
        return self._client.call("env.get_env_meta", timeout_s=_TIMEOUT_S["default"])

    def _planner_call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        ret = self._client.call(
            f"env.{method}",
            kwargs=kwargs,
            timeout_s=_TIMEOUT_S.get(f"env.{method}", _TIMEOUT_S["default"]),
        )
        if not isinstance(ret, dict):
            raise RuntimeError(f"env.{method} returned non-dict result: {type(ret)!r}")
        return ret

    def observe(self, *, camera: str) -> dict[str, Any]:
        return self._planner_call("observe", camera=camera)

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
