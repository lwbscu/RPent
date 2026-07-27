"""BEHAVIOR env RPC client."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from copy import deepcopy
from typing import Any

import numpy as np

from robots.behavior.schemas import (
    ROTATE_WRIST_RUNTIME_TIMEOUT_S,
    validate_dashboard_manual_command,
    validate_relative_navigation_motion,
)
from rpent.utils.rpc import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 1800.0,
    "env.pi0_nav_pick_chunk_step": 1800.0,
    "env.prepare_vla_invocation": 120.0,
    "env.guard_tool_call": 30.0,
    "env.save_robot_state_checkpoint": 120.0,
    "env.finalize_paused_runtime": 120.0,
    "env.current_observation": 120.0,
    "env.observe": 120.0,
    "env.pixel_to_world": 120.0,
    "env.move_to": 1800.0,
    "env.navigate_to": 1800.0,
    "env.rotate_wrist": 1800.0,
    "env.close": 120.0,
    "env.open": 120.0,
    "env.press": 1800.0,
    "env.dashboard_control_capabilities": 30.0,
    "env.dashboard_manual_command": 360.0,
}
_SUCCESS_CLEANUP_RPC_METHODS = frozenset({"env.finalize_paused_runtime"})


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
        self._official_success_latched = False
        self._official_success_receipt: dict[str, Any] | None = None
        self._expected_run_nonce: str | None = None
        self._attempt_identity: tuple[str, str, int] | None = None
        server_meta = self._rpc_call(
            "env.get_env_meta",
            timeout_s=_TIMEOUT_S["default"],
        )
        if not isinstance(server_meta, dict):
            raise RuntimeError(f"env_meta must be a mapping, got {type(server_meta)!r}")
        mismatches = {
            key: {"expected": expected, "actual": server_meta.get(key)}
            for key, expected in expected_meta.items()
            if server_meta.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"env_meta mismatch: {mismatches!r}")
        self.server_meta = dict(server_meta)
        run_nonce = server_meta.get("run_nonce")
        if run_nonce is not None:
            if (
                not isinstance(run_nonce, str)
                or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
            ):
                raise RuntimeError(
                    "env_meta.run_nonce must be 32 lowercase hex characters"
                )
            self._expected_run_nonce = run_nonce

    def reset(self) -> tuple[dict[str, Any], Any]:
        """Runtime-only initialization for the current fresh episode."""

        ret = self._rpc_call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        info = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else None
        self._bind_attempt_identity_from_info(info)
        if not self._official_success_latched:
            self.episode_done = False
        self.total_env_steps = 0
        return ret

    @staticmethod
    def _raw_success(info: Any) -> bool:
        done = info.get("done") if isinstance(info, dict) else None
        value = done.get("success") if isinstance(done, dict) else None
        return isinstance(value, (bool, np.bool_)) and bool(value)

    @staticmethod
    def _receipt_from_info(info: Any) -> dict[str, Any] | None:
        runtime = info.get("_rpent") if isinstance(info, dict) else None
        if not isinstance(runtime, dict):
            return None
        receipt = runtime.get("official_success_receipt")
        if isinstance(receipt, dict):
            return deepcopy(receipt)
        monitor = runtime.get("pi0_nav_pick_monitor")
        receipt = (
            monitor.get("official_success_receipt")
            if isinstance(monitor, dict)
            else None
        )
        return deepcopy(receipt) if isinstance(receipt, dict) else None

    @staticmethod
    def _canonical_receipt_bytes(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def _bind_attempt_identity_from_info(self, info: Any) -> None:
        runtime = info.get("_rpent") if isinstance(info, dict) else None
        if not isinstance(runtime, dict):
            return
        run_nonce = runtime.get("run_nonce")
        attempt_nonce = runtime.get("attempt_nonce")
        attempt_index = runtime.get("attempt_index")
        if run_nonce is None and attempt_nonce is None and attempt_index is None:
            return
        if (
            not isinstance(run_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
            or not isinstance(attempt_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", attempt_nonce) is None
            or isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 1
        ):
            raise RuntimeError("env.reset returned an invalid runtime attempt identity")
        expected_run_nonce = getattr(self, "_expected_run_nonce", None)
        if expected_run_nonce is not None and run_nonce != expected_run_nonce:
            raise RuntimeError("env.reset run_nonce disagrees with env metadata")
        identity = (run_nonce, attempt_nonce, attempt_index)
        bound = getattr(self, "_attempt_identity", None)
        if bound is not None and bound != identity:
            raise RuntimeError("env runtime attempt identity changed after binding")
        self._attempt_identity = identity

    def _valid_success_receipt(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        required = {
            "schema_version",
            "source",
            "run_nonce",
            "attempt_nonce",
            "attempt_index",
            "env_step",
            "raw_done",
            "receipt_sha256",
        }
        if set(value) != required:
            return None
        raw_done = value.get("raw_done")
        run_nonce = value.get("run_nonce")
        attempt_nonce = value.get("attempt_nonce")
        attempt_index = value.get("attempt_index")
        env_step = value.get("env_step")
        digest = value.get("receipt_sha256")
        if (
            value.get("schema_version") != 1
            or isinstance(value.get("schema_version"), bool)
            or value.get("source") != 'info["done"]["success"]'
            or not isinstance(raw_done, dict)
            or type(raw_done.get("success")) is not bool
            or raw_done["success"] is not True
            or not isinstance(run_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
            or not isinstance(attempt_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", attempt_nonce) is None
            or isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 1
            or isinstance(env_step, bool)
            or not isinstance(env_step, int)
            or env_step < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return None
        material = {key: item for key, item in value.items() if key != "receipt_sha256"}
        try:
            expected_digest = hashlib.sha256(
                self._canonical_receipt_bytes(material)
            ).hexdigest()
        except (TypeError, ValueError):
            return None
        if not hmac.compare_digest(digest, expected_digest):
            return None
        expected_run_nonce = getattr(self, "_expected_run_nonce", None)
        if expected_run_nonce is not None and run_nonce != expected_run_nonce:
            return None
        attempt_identity = getattr(self, "_attempt_identity", None)
        if attempt_identity is not None and attempt_identity != (
            run_nonce,
            attempt_nonce,
            attempt_index,
        ):
            return None
        return deepcopy(value)

    def _latch_success_response(self, ret: Any) -> None:
        info: Any = None
        direct_receipt: Any = None
        if isinstance(ret, (tuple, list)) and len(ret) == 5:
            info = ret[4]
        elif isinstance(ret, (tuple, list)) and len(ret) == 2:
            info = ret[1]
        elif isinstance(ret, dict):
            info = ret if isinstance(ret.get("done"), dict) else ret.get("info")
            direct_receipt = ret.get("official_success_receipt")
        receipt = self._valid_success_receipt(direct_receipt)
        if receipt is None:
            receipt = self._valid_success_receipt(self._receipt_from_info(info))
        raw_success = self._raw_success(info)
        if receipt is not None:
            if self._official_success_receipt is None:
                self._official_success_receipt = receipt
            elif self._official_success_receipt != receipt:
                raise RuntimeError(
                    "env official success receipt changed after the first latch"
                )
        if raw_success or receipt is not None:
            self._official_success_latched = True
            self.episode_done = True

    def _rpc_call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        if bool(getattr(self, "_official_success_latched", False)) and (
            method not in _SUCCESS_CLEANUP_RPC_METHODS
        ):
            raise RuntimeError(
                "raw task success is terminal; RPC rejected before transport"
            )
        ret = self._client.call(
            method,
            args=args,
            kwargs=kwargs,
            timeout_s=timeout_s,
        )
        self._latch_success_response(ret)
        return ret

    def prepare_vla_invocation(
        self,
        *,
        invocation_id: str,
        call_index: int,
        vla_status: dict[str, Any] | None,
        current_object_visual_check: dict[str, Any] | None = None,
        baseline_internal_authorization: bool = False,
    ) -> dict[str, Any]:
        return self._planner_call(
            "prepare_vla_invocation",
            invocation_id=invocation_id,
            call_index=call_index,
            vla_status=vla_status,
            current_object_visual_check=current_object_visual_check,
            baseline_internal_authorization=baseline_internal_authorization,
        )

    def guard_tool_call(
        self, *, name: str, input_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return self._planner_call(
            "guard_tool_call",
            name=name,
            input_dict=input_dict,
        )

    def _track_step_result(self, ret: tuple[Any, Any, Any, Any, Any]) -> None:
        _, _, term, trunc, info = ret
        if np.asarray(term).any() or np.asarray(trunc).any():
            self.episode_done = True
        if isinstance(info, dict):
            rpent = info.get("_rpent")
            if isinstance(rpent, dict) and "total_env_steps" in rpent:
                self.total_env_steps = int(rpent["total_env_steps"])
            if isinstance(rpent, dict) and "global_env_steps" in rpent:
                self.total_env_steps = int(rpent["global_env_steps"])

    def pi0_nav_pick_chunk_step(
        self,
        actions,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if self.episode_done:
            raise RuntimeError("env.pi0_nav_pick_chunk_step called after episode done")
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (32, 23):
            raise ValueError(
                f"pi0_nav_pick requires one complete [32,23] chunk, got {action_array.shape}"
            )
        ret = self._rpc_call(
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
        semantic_label: str | None = None,
        terminal_failure: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        kwargs = {} if semantic_label is None else {"semantic_label": semantic_label}
        if terminal_failure is not None:
            kwargs["terminal_failure"] = dict(terminal_failure)
        return self._planner_call("save_robot_state_checkpoint", **kwargs)

    def finalize_paused_runtime(self, vla_status: dict[str, Any]) -> dict[str, Any]:
        payload = dict(vla_status)
        if self.vla_endpoint:
            payload.setdefault("endpoint", self.vla_endpoint)
        ret = self._rpc_call(
            "env.finalize_paused_runtime",
            args=(payload,),
            timeout_s=_TIMEOUT_S["env.finalize_paused_runtime"],
        )
        if not isinstance(ret, dict):
            raise RuntimeError("env.finalize_paused_runtime returned a non-dict result")
        return ret

    def current_observation(self) -> tuple[dict[str, Any], Any]:
        """Refresh synchronized sensors without resetting or advancing physics."""
        ret = self._rpc_call(
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
        return observation, info

    def get_env_meta(self) -> dict[str, Any]:
        return self._rpc_call("env.get_env_meta", timeout_s=_TIMEOUT_S["default"])

    def _planner_call(
        self,
        method: str,
        *,
        _runtime_deadline_s: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        rpc_timeout_s = _TIMEOUT_S.get(f"env.{method}", _TIMEOUT_S["default"])
        requested_timeout = kwargs.get("timeout_s")
        if _runtime_deadline_s is not None and requested_timeout is not None:
            raise ValueError(
                "runtime deadline and public primitive timeout are mutually exclusive"
            )
        primitive_deadline_s = (
            _runtime_deadline_s
            if _runtime_deadline_s is not None
            else requested_timeout
        )
        if primitive_deadline_s is not None:
            # The primitive owns the hard deadline.  Keep a bounded transport
            # grace period for serializing its structured timeout result rather
            # than leaving every planner RPC blocked for the global 30 minutes.
            primitive_deadline_s = float(primitive_deadline_s)
            if not np.isfinite(primitive_deadline_s) or primitive_deadline_s <= 0.0:
                raise ValueError("runtime deadline must be finite and positive")
            rpc_timeout_s = min(
                rpc_timeout_s,
                max(30.0, primitive_deadline_s + 60.0),
            )
        ret = self._rpc_call(
            f"env.{method}",
            kwargs=kwargs,
            timeout_s=rpc_timeout_s,
        )
        if not isinstance(ret, dict):
            raise RuntimeError(f"env.{method} returned non-dict result: {type(ret)!r}")
        if "total_env_steps" in ret:
            self.total_env_steps = int(ret["total_env_steps"])
        if "global_env_steps" in ret:
            self.total_env_steps = int(ret["global_env_steps"])
        return ret

    def observe(
        self,
        *,
        camera: str,
        frame_review: dict[str, Any] | None = None,
        depth_probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if camera not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("camera must be head, left_wrist, or right_wrist")
        if frame_review is not None and depth_probe is not None:
            raise ValueError("frame_review and depth_probe are mutually exclusive")
        kwargs: dict[str, Any] = {"camera": camera}
        if frame_review is not None:
            kwargs["frame_review"] = frame_review
        if depth_probe is not None:
            kwargs["depth_probe"] = depth_probe
        return self._planner_call("observe", **kwargs)

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        """Return fail-closed simulator-owned manual-control capabilities."""

        return self._planner_call("dashboard_control_capabilities")

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        """Execute one server-sized manual command in one env-RPC transaction."""

        command = validate_dashboard_manual_command(
            target=target,
            action=action,
            camera=camera,
        )
        result = self._planner_call("dashboard_manual_command", **command)
        frames = result.get("_frames_bytes")
        capture_complete = bool(
            isinstance(frames, dict)
            and set(frames) == {"head", "left_wrist", "right_wrist"}
            and all(isinstance(value, bytes) for value in frames.values())
        )
        success_without_capture = bool(
            result.get("task_success") is True
            and self._valid_success_receipt(result.get("official_success_receipt"))
            is not None
            and isinstance(result.get("capture_error"), str)
            and bool(result["capture_error"])
        )
        if not capture_complete and not success_without_capture:
            raise RuntimeError(
                "env.dashboard_manual_command omitted the atomic three-camera capture"
            )
        if success_without_capture:
            return result
        group_id = result.get("capture_group_id")
        simulator_step = result.get("simulator_step")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("env.dashboard_manual_command omitted capture_group_id")
        if isinstance(simulator_step, bool) or not isinstance(
            simulator_step, (int, np.integer)
        ):
            raise RuntimeError("env.dashboard_manual_command omitted simulator_step")
        return result

    @staticmethod
    def _validated_analytic_hand(hand: str) -> str:
        if not isinstance(hand, str) or hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        return hand

    def move_to(
        self,
        *,
        hand: str,
        target: dict[str, Any],
        visual_hand_check: dict[str, Any],
        position_tolerance_m: float = 0.02,
        max_travel_m: float = 0.25,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        hand = self._validated_analytic_hand(hand)
        return self._planner_call(
            "move_to",
            hand=hand,
            target=target,
            visual_hand_check=visual_hand_check,
            position_tolerance_m=position_tolerance_m,
            max_travel_m=max_travel_m,
            timeout_s=timeout_s,
        )

    @staticmethod
    def _validated_navigation_visual_check(
        navigation_visual_check: Any,
    ) -> dict[str, str]:
        if not isinstance(navigation_visual_check, dict):
            raise ValueError("navigation_visual_check must be an object")
        required = {"camera", "frame_id", "assessment"}
        if set(navigation_visual_check) != required:
            raise ValueError(
                "navigation_visual_check requires exactly camera, frame_id, "
                "and assessment"
            )
        if navigation_visual_check["camera"] != "head":
            raise ValueError("navigation_visual_check.camera must be 'head'")
        frame_id = navigation_visual_check["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(
                "navigation_visual_check.frame_id must be a non-empty string"
            )
        if (
            navigation_visual_check["assessment"]
            != "navigation_target_visually_confirmed"
        ):
            raise ValueError(
                "navigation_visual_check.assessment must be "
                "'navigation_target_visually_confirmed'"
            )
        return {
            "camera": "head",
            "frame_id": frame_id.strip(),
            "assessment": "navigation_target_visually_confirmed",
        }

    @staticmethod
    def _validated_navigation_number(
        name: str,
        value: Any,
        *,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a finite number")
        number = float(value)
        lower_ok = number >= minimum if minimum_inclusive else number > minimum
        upper_ok = maximum is None or number <= maximum
        if not np.isfinite(number) or not lower_ok or not upper_ok:
            bounds = (
                f"[{minimum},{maximum}]"
                if minimum_inclusive
                else f"({minimum},{maximum}]"
            )
            raise ValueError(f"{name} must be finite and within {bounds}")
        return number

    def navigate_to(
        self,
        *,
        projection_id: str | None = None,
        navigation_visual_check: dict[str, Any] | None = None,
        relative_motion: dict[str, Any] | None = None,
        standoff_m: float | None = None,
        max_travel_m: float | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        if relative_motion is None:
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("projection_id must be a non-empty string")
            if navigation_visual_check is None:
                raise ValueError(
                    "navigation_visual_check is required for projection navigation"
                )
            payload = {
                "projection_id": projection_id.strip(),
                "navigation_visual_check": (
                    self._validated_navigation_visual_check(navigation_visual_check)
                ),
                "standoff_m": self._validated_navigation_number(
                    "standoff_m",
                    0.85 if standoff_m is None else standoff_m,
                    minimum=0.45,
                    maximum=1.50,
                ),
                "max_travel_m": self._validated_navigation_number(
                    "max_travel_m",
                    1.0 if max_travel_m is None else max_travel_m,
                    minimum=0.0,
                    maximum=1.50,
                    minimum_inclusive=False,
                ),
            }
        else:
            if any(
                value is not None
                for value in (
                    projection_id,
                    navigation_visual_check,
                    standoff_m,
                    max_travel_m,
                )
            ):
                raise ValueError(
                    "relative_motion is mutually exclusive with projection "
                    "navigation arguments"
                )
            payload = {
                "relative_motion": validate_relative_navigation_motion(relative_motion)
            }
        payload["timeout_s"] = self._validated_navigation_number(
            "timeout_s",
            timeout_s,
            minimum=0.0,
            minimum_inclusive=False,
        )
        return self._planner_call("navigate_to", **payload)

    def rotate_wrist(
        self,
        *,
        hand: str,
        relative_axis_angle: list[float],
        visual_hand_check: dict[str, Any],
        frame: str = "eef",
    ) -> dict[str, Any]:
        hand = self._validated_analytic_hand(hand)
        return self._planner_call(
            "rotate_wrist",
            _runtime_deadline_s=ROTATE_WRIST_RUNTIME_TIMEOUT_S,
            hand=hand,
            relative_axis_angle=relative_axis_angle,
            frame=frame,
            visual_hand_check=visual_hand_check,
        )

    def close(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        hand = self._validated_analytic_hand(hand)
        return self._planner_call(
            "close",
            hand=hand,
            visual_hand_check=visual_hand_check,
            timeout_s=timeout_s,
        )

    def open(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        release_visual_check: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        hand = self._validated_analytic_hand(hand)
        kwargs: dict[str, Any] = {
            "hand": hand,
            "visual_hand_check": visual_hand_check,
            "timeout_s": timeout_s,
        }
        if release_visual_check is not None:
            kwargs["release_visual_check"] = release_visual_check
        return self._planner_call("open", **kwargs)

    def press(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        projection_id: str,
        travel_m: float,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        hand = self._validated_analytic_hand(hand)
        return self._planner_call(
            "press",
            hand=hand,
            visual_hand_check=visual_hand_check,
            projection_id=projection_id,
            travel_m=travel_m,
            timeout_s=timeout_s,
        )

    def pixel_to_world(
        self,
        *,
        camera: str,
        frame_id: str,
        u: int,
        v: int,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        if camera not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("camera must be head, left_wrist, or right_wrist")
        return self._planner_call(
            "pixel_to_world",
            camera=camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
        )

    def close_transport(self) -> None:
        """Close only the client transport; runtime ownership stays with provider."""
        self._client.close()
