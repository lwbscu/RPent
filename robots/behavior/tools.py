"""Closed BEHAVIOR primitives for full-task, planner, and local VLA modes."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.prepress import (
    NEGATIVE_CASE_TO_FACE_CLASS,
    PREPRESS_AXIAL_STANDOFF_MAX_M,
    PREPRESS_AXIAL_STANDOFF_MIN_M,
    PRESS_STAGING_AXIAL_STANDOFF_MAX_M,
)
from robots.behavior.run_manifest import MANIFEST_FILENAME
from robots.behavior.schemas import (
    DEFAULT_ACTION_CHUNK,
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    TEMP_STATE_CHECKPOINT_PATTERN,
    extract_policy_state,
    segment_ranges,
    validate_action_chunk,
)
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("behavior")

_PI0_NAV_PICK_MONITOR_FIELDS = (
    "executed_steps",
    "handoff_env_steps",
    "total_env_steps",
    "local_grasp_success",
    "held_hand",
    "per_hand",
    "current_criteria",
    "validator_trace_path",
    "state_checkpoint_path",
    "handoff_state",
    "action_source",
    "vla_actions_enabled",
    "paused_runtime_path",
)
_PI0_NAV_PICK_OPTIONAL_MONITOR_FIELDS = (
    "visual_review",
    "stop_reason",
    "strict_local_grasp_success",
    "usable_post_pick_saved",
    "save_policy",
    "warnings",
)
_TEMP_STATE_CHECKPOINT_NAME = re.compile(TEMP_STATE_CHECKPOINT_PATTERN)


def _is_temporary_checkpoint(name: str) -> bool:
    return bool(_TEMP_STATE_CHECKPOINT_NAME.fullmatch(name))


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _as_bool(value: Any) -> bool:
    array = np.asarray(_jsonable(value))
    return bool(array.any()) if array.size else False


def official_task_success(info: Any) -> bool:
    """Read only the official raw BEHAVIOR success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    return bool(done.get("success", False)) if isinstance(done, dict) else False


def _public_info_summary(info: Any) -> Any:
    """Expose only stop/accounting fields, never simulator observation metadata."""
    if not isinstance(info, dict):
        return {}
    return {key: _jsonable(info[key]) for key in ("done", "_rpent") if key in info}


def _public_validator_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy only RGB and proprio fields into the local-grasp validator."""
    if not isinstance(obs, dict):
        raise TypeError("local grasp validator observation must be a mapping")
    required = ("main_images", "wrist_images", "states")
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"local grasp validator observation missing {missing}")
    return {key: np.asarray(obs[key]).copy() for key in required}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


def _png_bytes(image: Any) -> bytes:
    from PIL import Image

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"visual review image must be HxWxC, got {array.shape}")
    buffer = BytesIO()
    Image.fromarray(array[..., :3], mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class _FullTaskRunner:
    """Run Pi0.5 chunks until an official or episode stop condition."""

    def __init__(
        self,
        *,
        env: Any,
        model: Any,
        max_episode_steps: int,
        output_dir: str | Path | None = None,
        video_path: str | Path | None = None,
        action_horizon: int = DEFAULT_ACTION_CHUNK,
        max_chunks: int | None = None,
    ) -> None:
        self.env = env
        self.model = model
        self.max_episode_steps = int(max_episode_steps)
        self.action_horizon = int(action_horizon)
        self.max_chunks = max_chunks
        configured_output = Path(output_dir) if output_dir else get_output_dir()
        # Direct unit construction may happen before logging initializes the
        # process output directory.
        self.output_dir = configured_output or Path.cwd()
        self.video_path = (
            Path(video_path) if video_path else self.output_dir / "episode.mp4"
        )
        self.last_result: dict[str, Any] | None = None

    def _prepare_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "states.json",
            "raw_final_info.json",
            "behavior_full_task_result.json",
        ):
            path = self.output_dir / name
            if path.exists():
                path.unlink()

    def _state_record(
        self,
        *,
        chunk: int,
        env_steps: int,
        obs: dict[str, Any],
        info: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        model_info: Any = None,
    ) -> dict[str, Any]:
        raw_proprio = np.asarray(obs["states"], dtype=np.float32)
        compact = extract_policy_state(raw_proprio)
        task_success = official_task_success(info)
        return {
            "chunk": int(chunk),
            "env_steps": int(env_steps),
            "task_language": str(obs.get("task_descriptions") or ""),
            "raw_proprio": _jsonable(raw_proprio),
            "policy_state": _jsonable(compact),
            "policy_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
            "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
            "reward": _jsonable(reward),
            "terminated": _as_bool(terminated),
            "truncated": _as_bool(truncated),
            "info": _jsonable(info),
            "model": _jsonable(model_info),
            "success": task_success,
            "task_success": task_success,
            "official_success_source": 'info["done"]["success"]',
        }

    def _persist_states(self, states: list[dict[str, Any]]) -> Path:
        path = self.output_dir / "states.json"
        _write_json_atomic(path, states)
        return path

    def run_full_task(self) -> dict[str, Any]:
        """Execute one complete task; this method intentionally takes no args."""
        self._prepare_output()
        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        env_steps_used = 0
        task_success = False
        terminated = False
        truncated = False
        stop_reason = "not_started"
        last_reward: Any = None
        last_info: Any = None
        error: str | None = None

        try:
            obs, info = self.env.reset()
            last_info = info
            initial = self._state_record(
                chunk=0,
                env_steps=0,
                obs=obs,
                info=info,
                reward=None,
                terminated=False,
                truncated=False,
            )
            states.append(initial)
            task_success = initial["task_success"]
            stop_reason = "task_success" if task_success else "running"
            self._persist_states(states)

            while (
                not task_success
                and not terminated
                and not truncated
                and (self.max_chunks is None or chunks_used < self.max_chunks)
                and env_steps_used < self.max_episode_steps
            ):
                actions, model_info = self.model.predict_action_batch(obs, mode="eval")
                actions = validate_action_chunk(
                    actions,
                    max_horizon=self.action_horizon,
                )
                remaining = self.max_episode_steps - env_steps_used
                actions = actions[:remaining]
                obs, reward, term, trunc, info = self.env.chunk_step(actions)

                chunks_used += 1
                executed_steps = (
                    info.get("_rpent", {}).get("executed_steps", actions.shape[0])
                    if isinstance(info, dict)
                    else actions.shape[0]
                )
                if (
                    isinstance(executed_steps, bool)
                    or int(executed_steps) != executed_steps
                    or not 1 <= int(executed_steps) <= actions.shape[0]
                ):
                    raise RuntimeError(
                        "invalid env executed_steps for run_full_task: "
                        f"{executed_steps!r}"
                    )
                env_steps_used += int(executed_steps)
                last_reward = reward
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)
                state = self._state_record(
                    chunk=chunks_used,
                    env_steps=env_steps_used,
                    obs=obs,
                    info=info,
                    reward=reward,
                    terminated=term,
                    truncated=trunc,
                    model_info=model_info,
                )
                states.append(state)
                task_success = state["task_success"]
                if task_success:
                    stop_reason = "task_success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                elif env_steps_used >= self.max_episode_steps:
                    stop_reason = "horizon"
                elif self.max_chunks is not None and chunks_used >= self.max_chunks:
                    stop_reason = "chunk_limit"
                self._persist_states(states)

            if stop_reason == "running":
                stop_reason = (
                    "horizon"
                    if env_steps_used >= self.max_episode_steps
                    else "chunk_limit"
                )
        except Exception as exc:
            logger.exception("run_full_task failed")
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "error"

        if states:
            states[-1]["success"] = bool(task_success)
            states[-1]["task_success"] = bool(task_success)
            states[-1]["stop_reason"] = stop_reason
        states_path = self._persist_states(states)
        raw_final_info_path = self.output_dir / "raw_final_info.json"
        _write_json_atomic(raw_final_info_path, _jsonable(last_info))
        result = {
            "_finish": True,
            "name": "run_full_task",
            "success": bool(task_success),
            "task_success": bool(task_success),
            "official_success_source": 'info["done"]["success"]',
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "chunks_used": chunks_used,
            "env_steps_used": env_steps_used,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "elapsed_s": round(time.time() - started, 2),
            "states_path": str(states_path),
            "raw_final_info_path": str(raw_final_info_path),
            "action_trace_path": str(self.output_dir / "behavior_action_trace.jsonl"),
            "manifest_path": str(self.output_dir / MANIFEST_FILENAME),
            "video_path": str(self.video_path),
            "video_fps": 15,
            "video_sample_every_env_steps": 4,
            "last_reward": _jsonable(last_reward),
            "last_info": _public_info_summary(last_info),
            "error": error,
        }
        _write_json_atomic(self.output_dir / "behavior_full_task_result.json", result)
        self.last_result = result
        return result


class BehaviorPrimitives:
    """Public BEHAVIOR primitive handlers shared by the isolated tool modes.

    The class is the only public implementation surface consumed by
    :class:`BehaviorToolkit`.  It deliberately delegates planner commands to
    the injected backend and keeps the full-task runner private.
    """

    def __init__(
        self,
        *,
        env: Any = None,
        model: Any = None,
        planner_backend: Any = None,
        max_episode_steps: int | None = None,
        output_dir: str | Path | None = None,
        video_path: str | Path | None = None,
        action_horizon: int = DEFAULT_ACTION_CHUNK,
        initial_observation: dict[str, Any] | None = None,
        initial_info: Any = None,
        local_grasp_validator: (
            Callable[[dict[str, Any], dict[str, Any]], bool] | None
        ) = None,
        hybrid_mode: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.planner_backend = planner_backend
        self.max_episode_steps = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.action_horizon = int(action_horizon)
        configured_output = Path(output_dir) if output_dir else get_output_dir()
        # Planner-only construction does not need artifacts and can occur in
        # unit tests before logging initializes the process output directory.
        self.output_dir = configured_output or Path.cwd()
        self.video_path = (
            Path(video_path) if video_path else self.output_dir / "episode.mp4"
        )
        self._current_observation = initial_observation
        self._current_info = initial_info
        self._local_grasp_validator = local_grasp_validator
        self._hybrid_mode = bool(hybrid_mode)
        self._pi0_navigate_to_segment_counter = 0
        self.last_result: dict[str, Any] | None = None
        self._full_task_runner: _FullTaskRunner | None = None
        if env is not None and model is not None and max_episode_steps is not None:
            self._full_task_runner = _FullTaskRunner(
                env=env,
                model=model,
                max_episode_steps=max_episode_steps,
                output_dir=self.output_dir,
                video_path=self.video_path,
                action_horizon=self.action_horizon,
            )

    def run_full_task(self) -> dict[str, Any]:
        """Explicitly delegate the only full-task tool to the private runner."""
        if self._full_task_runner is None:
            raise RuntimeError(
                "run_full_task requires env, model, and max_episode_steps"
            )
        result = self._full_task_runner.run_full_task()
        self.last_result = result
        return result

    def _planner_method(self, name: str) -> Callable[..., dict[str, Any]]:
        if self.planner_backend is None:
            raise RuntimeError(f"{name} requires a planner_backend")
        method = getattr(self.planner_backend, name, None)
        if not callable(method):
            raise RuntimeError(f"planner_backend does not implement {name}")
        return method

    def observe(self, *, camera: str) -> dict[str, Any]:
        if self.planner_backend is not None:
            return self._planner_method("observe")(camera=camera)
        return self._public_checkpoint_result(
            name="observe",
            result=self._env_method("observe")(camera=camera),
        )

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
        kwargs = {
            "camera": camera,
            "frame_id": frame_id,
            "u": u,
            "v": v,
            "depth_window_px": depth_window_px,
            "output_frame": output_frame,
        }
        if self.planner_backend is not None:
            return self._planner_method("pixel_to_world")(**kwargs)
        return self._public_checkpoint_result(
            name="pixel_to_world",
            result=self._env_method("pixel_to_world")(**kwargs),
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
        return self._planner_method("navigate_to")(
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
        return self._planner_method("move_to")(
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
        return self._planner_method("pick")(
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
        return self._planner_method("rotate_wrist")(
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
        return self._planner_method("press")(
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
        return self._planner_method("release")(
            hand=hand,
            opening=opening,
            retreat_vector=retreat_vector,
            retreat_m=retreat_m,
            timeout_s=timeout_s,
        )

    def _env_method(self, name: str) -> Callable[..., dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(f"{name} requires an active BEHAVIOR env")
        method = getattr(self.env, name, None)
        if not callable(method):
            raise RuntimeError(f"env does not implement {name}")
        return method

    def _robot_checkpoint_method(self, name: str) -> Callable[..., dict[str, Any]]:
        return self._env_method(name)

    def _public_checkpoint_result(
        self,
        *,
        name: str,
        result: Any,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError(f"{name} returned a non-mapping result")
        public = dict(result)
        for field in ("primitive_success", "task_success", "stop_reason"):
            if field not in public:
                raise RuntimeError(f"{name} result omitted {field}")
        if not isinstance(public["primitive_success"], bool):
            raise RuntimeError(f"{name} primitive_success must be boolean")
        if not isinstance(public["task_success"], bool):
            raise RuntimeError(f"{name} task_success must be boolean")
        if not isinstance(public["stop_reason"], str) or not public["stop_reason"]:
            raise RuntimeError(f"{name} stop_reason must be non-empty")
        info = public.get("info")
        if isinstance(info, dict):
            official = official_task_success(info)
            if public["task_success"] is not official:
                raise RuntimeError(
                    f"{name} task_success disagrees with info.done.success"
                )
        elif public.get("official_success_source") != 'info["done"]["success"]':
            raise RuntimeError(
                f"{name} result lacks verifiable official success provenance"
            )
        public.update(
            {
                "_finish": False,
                "name": name,
                "official_success_source": 'info["done"]["success"]',
            }
        )
        self.last_result = public
        return public

    def inspect_post_pick_state(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_1",
    ) -> dict[str, Any]:
        """Bind dynamic held/press roles from a real post-pick checkpoint."""

        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise ValueError("checkpoint_name must be a non-empty string")
        if checkpoint_name != "state_checkpoint_1":
            raise ValueError("checkpoint save only accepts state_checkpoint_1")
        return self._public_checkpoint_result(
            name="inspect_post_pick_state",
            result=self._env_method("inspect_post_pick_state")(
                checkpoint_name=checkpoint_name,
            ),
        )

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
        """Submit one strict, frame-bound button visibility decision."""

        if camera not in {"head", "held_wrist", "press_wrist"}:
            raise ValueError("camera must be head, held_wrist, or press_wrist")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(button_visible, bool):
            raise ValueError("button_visible must be boolean")
        if (
            negative_case is not None
            and negative_case not in NEGATIVE_CASE_TO_FACE_CLASS
        ):
            allowed = ", ".join(sorted(NEGATIVE_CASE_TO_FACE_CLASS))
            raise ValueError(f"negative_case must be null or one of: {allowed}")
        if button_visible:
            required_signature = {
                "red_front_face",
                "black_round_or_oval_disk",
                "white_outer_ring",
                "red_center_bump",
            }
            if (
                not isinstance(positive_signature, dict)
                or set(positive_signature) != required_signature
                or not all(value is True for value in positive_signature.values())
            ):
                raise ValueError(
                    "visible button requires the complete positive_signature"
                )
            if negative_case is not None:
                raise ValueError("visible button may not declare a negative_case")
            bbox = np.asarray(bbox_xyxy, dtype=np.float64)
            center = np.asarray(center_uv, dtype=np.float64)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                raise ValueError("visible button requires finite bbox_xyxy[4]")
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError("visible button requires finite center_uv[2]")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("bbox_xyxy must have positive width and height")
            if not (
                bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3]
            ):
                raise ValueError("center_uv must lie inside bbox_xyxy")
        else:
            if negative_case not in NEGATIVE_CASE_TO_FACE_CLASS:
                allowed = ", ".join(sorted(NEGATIVE_CASE_TO_FACE_CLASS))
                raise ValueError(f"NOT_VISIBLE negative_case must be one of: {allowed}")
            if any(
                value is not None
                for value in (positive_signature, bbox_xyxy, center_uv)
            ):
                raise ValueError(
                    "NOT_VISIBLE must not include positive signature or coordinates"
                )
        return self._public_checkpoint_result(
            name="declare_button_visibility",
            result=self._env_method("declare_button_visibility")(
                camera=camera,
                frame_id=frame_id,
                button_visible=button_visible,
                positive_signature=positive_signature,
                negative_case=negative_case,
                bbox_xyxy=bbox_xyxy,
                center_uv=center_uv,
            ),
        )

    def project_button(
        self,
        *,
        gate_id: str,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        """Project only a button center authorized by a positive hard gate."""

        if not isinstance(gate_id, str) or not gate_id.strip():
            raise ValueError("gate_id must be a non-empty string")
        if (
            isinstance(depth_window_px, bool)
            or not isinstance(depth_window_px, (int, np.integer))
            or int(depth_window_px) < 1
        ):
            raise ValueError("depth_window_px must be a positive integer")
        return self._public_checkpoint_result(
            name="project_button",
            result=self._env_method("project_button")(
                gate_id=gate_id,
                depth_window_px=int(depth_window_px),
            ),
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
        """Evaluate geometry without moving either hand or pressing."""

        if not isinstance(projection_id, str) or not projection_id.strip():
            raise ValueError("projection_id must be a non-empty string")
        values = {
            "max_line_distance_m": max_line_distance_m,
            "max_opposition_angle_deg": max_opposition_angle_deg,
            "min_axial_standoff_m": min_axial_standoff_m,
            "max_axial_standoff_m": max_axial_standoff_m,
        }
        numeric: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{name} must be finite")
            numeric[name] = number
        if numeric["max_line_distance_m"] <= 0.0:
            raise ValueError("max_line_distance_m must be positive")
        if numeric["max_opposition_angle_deg"] <= 0.0:
            raise ValueError("max_opposition_angle_deg must be positive")
        if numeric["min_axial_standoff_m"] < 0.0:
            raise ValueError("min_axial_standoff_m must be non-negative")
        if numeric["max_axial_standoff_m"] <= numeric["min_axial_standoff_m"]:
            raise ValueError("max_axial_standoff_m must exceed min_axial_standoff_m")
        return self._public_checkpoint_result(
            name="evaluate_prepress_geometry",
            result=self._env_method("evaluate_prepress_geometry")(
                projection_id=projection_id,
                **numeric,
            ),
        )

    def prepress_move_to(
        self,
        *,
        role: str = "held",
        button_goal: dict[str, Any],
        plan_only: bool = False,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        """Resolve a button-space goal into a role-bound CuRobo motion."""

        if role not in {"held", "press"}:
            raise ValueError("role must be 'held' or 'press'")
        if not isinstance(button_goal, dict):
            raise ValueError("button_goal must be an object")
        goal = dict(button_goal)
        kind = goal.get("kind")
        expected_kind = "held_button_alignment" if role == "held" else "press_staging"
        if kind != expected_kind:
            raise ValueError(
                f"role={role!r} requires button_goal.kind={expected_kind!r}"
            )
        if role == "held":
            if goal.get("head_view") != "side" or goal.get("face_toward") != "press":
                raise ValueError(
                    "held_button_alignment requires head_view='side' and "
                    "face_toward='press'"
                )
            numeric_bounds = {
                "toward_robot_m": (0.0, 0.30, None),
                "side_view_tolerance_deg": (0.0, 30.0, 15.0),
                "face_toward_tolerance_deg": (0.0, 45.0, 30.0),
                "position_slack_m": (0.0, 0.10, 0.04),
                "minimum_table_clearance_m": (0.08, 0.25, 0.12),
            }
            for name, (lower, upper, default) in numeric_bounds.items():
                value = goal.get(name, default)
                if value is None or isinstance(value, bool):
                    raise ValueError(f"button_goal.{name} must be finite")
                number = float(value)
                if not np.isfinite(number) or number < lower or number > upper:
                    raise ValueError(
                        f"button_goal.{name} must be in [{lower}, {upper}]"
                    )
                if name.endswith("tolerance_deg") and number <= 0.0:
                    raise ValueError(f"button_goal.{name} must be positive")
                goal[name] = number
            head_target_uv = goal.get("head_target_uv")
            if head_target_uv is not None:
                values = np.asarray(head_target_uv, dtype=np.float64).reshape(-1)
                if values.size != 2 or not np.isfinite(values).all():
                    raise ValueError(
                        "button_goal.head_target_uv must contain two finite pixels"
                    )
                goal["head_target_uv"] = values.tolist()
                radius_px = goal.get("head_target_radius_px", 60.0)
                if isinstance(radius_px, bool):
                    raise ValueError("button_goal.head_target_radius_px must be finite")
                radius_px = float(radius_px)
                if not np.isfinite(radius_px) or not 0.0 <= radius_px <= 160.0:
                    raise ValueError(
                        "button_goal.head_target_radius_px must be in [0, 160]"
                    )
                goal["head_target_radius_px"] = radius_px
            elif "head_target_radius_px" in goal:
                raise ValueError(
                    "button_goal.head_target_radius_px requires head_target_uv"
                )
            alignment_phase = goal.get("alignment_phase", "joint")
            if alignment_phase not in {"joint", "position_first", "normal_refine"}:
                raise ValueError(
                    "button_goal.alignment_phase must be joint, position_first, "
                    "or normal_refine"
                )
            if alignment_phase == "position_first" and head_target_uv is None:
                raise ValueError(
                    "button_goal.alignment_phase=position_first requires head_target_uv"
                )
            goal["alignment_phase"] = alignment_phase
            goal.setdefault("candidate_budget", 12)
        else:
            projection_id = goal.get("projection_id")
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("press_staging requires projection_id")
            alignment_phase = goal.get("alignment_phase", "final")
            if alignment_phase not in {"final", "observation"}:
                raise ValueError(
                    "button_goal.alignment_phase must be final or observation"
                )
            standoff = goal.get("standoff_m", 0.055)
            if isinstance(standoff, bool):
                raise ValueError("button_goal.standoff_m must be finite")
            standoff = float(standoff)
            standoff_max = (
                PREPRESS_AXIAL_STANDOFF_MAX_M
                if alignment_phase == "final"
                else PRESS_STAGING_AXIAL_STANDOFF_MAX_M
            )
            if (
                not np.isfinite(standoff)
                or not PREPRESS_AXIAL_STANDOFF_MIN_M <= standoff <= standoff_max
            ):
                raise ValueError(
                    "button_goal.standoff_m must be in "
                    f"[0.03, {standoff_max:.2f}] for {alignment_phase} alignment"
                )
            goal["alignment_phase"] = alignment_phase
            goal["standoff_m"] = standoff
            goal.setdefault("candidate_budget", 8)
        budget = goal.get("candidate_budget")
        max_budget = 32 if role == "held" else 16
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
            raise ValueError("button_goal.candidate_budget must be an integer")
        if not 1 <= int(budget) <= max_budget:
            raise ValueError(
                f"button_goal.candidate_budget must be in [1, {max_budget}]"
            )
        goal["candidate_budget"] = int(budget)
        if not isinstance(plan_only, bool):
            raise ValueError("plan_only must be boolean")
        if isinstance(timeout_s, bool):
            raise ValueError("timeout_s must be finite and positive")
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        return self._public_checkpoint_result(
            name="move_to",
            result=self._env_method("prepress_move_to")(
                role=role,
                button_goal=goal,
                plan_only=plan_only,
                timeout_s=timeout,
            ),
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
        """Rotate a dynamically bound checkpoint wrist in post-pick mode."""

        if role not in {"held", "press"}:
            raise ValueError("role must be 'held' or 'press'")

        if (target_quat_xyzw is None) == (relative_axis_angle is None):
            raise ValueError(
                "rotate_wrist requires exactly one of target_quat_xyzw or "
                "relative_axis_angle"
            )
        if frame not in {"world", "eef"}:
            raise ValueError("frame must be 'world' or 'eef'")
        if not isinstance(plan_only, bool):
            raise ValueError("plan_only must be boolean")
        if isinstance(timeout_s, bool):
            raise ValueError("timeout_s must be finite and positive")
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        return self._public_checkpoint_result(
            name="rotate_wrist",
            result=self._env_method("prepress_rotate_wrist")(
                role=role,
                target_quat_xyzw=target_quat_xyzw,
                relative_axis_angle=relative_axis_angle,
                frame=frame,
                plan_only=plan_only,
                timeout_s=timeout,
            ),
        )

    def save_post_pick_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_2",
        stage: str = "pre_press_alignment",
        visual_review: bool = True,
    ) -> dict[str, Any]:
        """Commit checkpoint2 through the fail-closed pre-press finalizer."""

        temporary = _is_temporary_checkpoint(checkpoint_name)
        if checkpoint_name != "state_checkpoint_2" and not temporary:
            raise ValueError(
                "post-pick checkpoint save accepts state_checkpoint_2 or "
                "tmp_state_checkpoint_*"
            )
        expected_stage = (
            "temporary_restore_point" if temporary else "pre_press_alignment"
        )
        if stage != expected_stage:
            raise ValueError(f"checkpoint stage must be {expected_stage}")
        if visual_review is not True:
            raise ValueError("checkpoint save requires visual_review=true")
        if temporary:
            context = self._env_method("inspect_post_pick_state")(
                checkpoint_name="state_checkpoint_1"
            )
            return self._public_checkpoint_result(
                name="save_robot_state_checkpoint",
                result=self._env_method("save_robot_state_checkpoint")(
                    checkpoint_name=checkpoint_name,
                    stage=stage,
                    held_hand=context["held_hand"],
                    press_hand=context["press_hand"],
                    object_name=context["object_name"],
                    require_current_grasp=True,
                    visual_review=True,
                ),
            )
        return self._public_checkpoint_result(
            name="save_robot_state_checkpoint",
            result=self._env_method("save_prepress_checkpoint")(
                checkpoint_name=checkpoint_name,
                stage=stage,
                visual_review=True,
            ),
        )

    def stage3_press_call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Route one allowlisted stage-3 composite to the active environment."""

        allowed = {
            "post_pick_close_press_gripper",
            "inspect_toggle_geometry",
            "post_pick_recenter_held_button",
            "post_pick_direct_finger_toggle",
            "post_success_hold_frames",
        }
        if name not in allowed:
            raise ValueError(f"unknown stage-3 press handler: {name}")
        return self._public_checkpoint_result(
            name=name,
            result=self._env_method(name)(**kwargs),
        )

    def save_prepress_checkpoint(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_2",
        stage: str = "pre_press_alignment",
        visual_review: bool = True,
    ) -> dict[str, Any]:
        """Save the geometry-gated pre-press robot-motion checkpoint."""

        for field, value in (
            ("checkpoint_name", checkpoint_name),
            ("stage", stage),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if checkpoint_name != "state_checkpoint_2" or stage != "pre_press_alignment":
            raise ValueError(
                "pre-press checkpoint save only accepts state_checkpoint_2 "
                "at stage pre_press_alignment"
            )
        if not isinstance(visual_review, bool):
            raise ValueError("visual_review must be boolean")
        return self._public_checkpoint_result(
            name="save_prepress_checkpoint",
            result=self._env_method("save_prepress_checkpoint")(
                checkpoint_name=checkpoint_name,
                stage=stage,
                visual_review=visual_review,
            ),
        )

    def save_robot_state_checkpoint(
        self,
        *,
        held_hand: str,
        press_hand: str,
        checkpoint_name: str = "state_checkpoint_1",
        stage: str = "post_pi0_nav_pick",
        object_name: str = "radio",
        require_current_grasp: bool = True,
        visual_review: bool = True,
    ) -> dict[str, Any]:
        """Save a robot-control checkpoint, never a simulator snapshot."""

        if held_hand not in {"left", "right"}:
            raise ValueError("held_hand must be 'left' or 'right'")
        if press_hand not in {"left", "right"}:
            raise ValueError("press_hand must be 'left' or 'right'")
        if held_hand == press_hand:
            raise ValueError("held_hand and press_hand must be different")
        for field, value in (
            ("checkpoint_name", checkpoint_name),
            ("stage", stage),
            ("object_name", object_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if checkpoint_name != "state_checkpoint_1" and not _is_temporary_checkpoint(
            checkpoint_name
        ):
            raise ValueError(
                "checkpoint save accepts state_checkpoint_1 or tmp_state_checkpoint_*"
            )
        if (
            _is_temporary_checkpoint(checkpoint_name)
            and stage != "temporary_restore_point"
        ):
            raise ValueError(
                "temporary checkpoint stage must be temporary_restore_point"
            )
        if not isinstance(require_current_grasp, bool):
            raise ValueError("require_current_grasp must be boolean")
        if not isinstance(visual_review, bool):
            raise ValueError("visual_review must be boolean")
        result = self._robot_checkpoint_method("save_robot_state_checkpoint")(
            checkpoint_name=checkpoint_name,
            stage=stage,
            held_hand=held_hand,
            press_hand=press_hand,
            object_name=object_name,
            require_current_grasp=require_current_grasp,
            visual_review=visual_review,
        )
        return self._public_checkpoint_result(
            name="save_robot_state_checkpoint",
            result=result,
        )

    def restore_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str = "state_checkpoint_1",
        checkpoint_path: str | None = None,
        mode: str = "plan_and_execute",
        keep_held_gripper_closed: bool = True,
        require_object_still_held: bool = True,
        timeout_s: float = 90,
    ) -> dict[str, Any]:
        """Execute a guarded cuRobo return to a robot-control checkpoint."""

        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise ValueError("checkpoint_name must be a non-empty string")
        if checkpoint_name not in {"state_checkpoint_1", "state_checkpoint_2"} and not (
            _is_temporary_checkpoint(checkpoint_name)
        ):
            raise ValueError(
                "checkpoint restore accepts state_checkpoint_1, "
                "state_checkpoint_2, or tmp_state_checkpoint_*"
            )
        if checkpoint_path is not None and (
            not isinstance(checkpoint_path, str) or not checkpoint_path.strip()
        ):
            raise ValueError("checkpoint_path must be null or a non-empty string")
        if mode != "plan_and_execute":
            raise ValueError("mode must be 'plan_and_execute'")
        if keep_held_gripper_closed is not True:
            raise ValueError("keep_held_gripper_closed must remain true")
        if require_object_still_held is not True:
            raise ValueError("require_object_still_held must remain true")
        if isinstance(timeout_s, bool):
            raise ValueError("timeout_s must be finite and positive")
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        result = self._robot_checkpoint_method("restore_robot_state_checkpoint")(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            mode=mode,
            keep_held_gripper_closed=keep_held_gripper_closed,
            require_object_still_held=require_object_still_held,
            timeout_s=timeout,
        )
        return self._public_checkpoint_result(
            name="restore_robot_state_checkpoint",
            result=result,
        )

    @staticmethod
    def _selected_gripper_opening(obs: dict[str, Any], hand: str) -> tuple[float, Any]:
        raw_proprio = np.asarray(obs["states"], dtype=np.float32)
        compact = extract_policy_state(raw_proprio)
        segment = POLICY_STATE_SEGMENTS[f"{hand}_gripper"]
        selected = compact[segment]
        if selected.shape != (1,):
            raise RuntimeError(
                f"selected {hand} compact gripper must contain one value, "
                f"got {selected.shape}"
            )
        return float(selected[0]), compact

    def _pi0_state_record(
        self,
        *,
        chunk: int,
        env_steps: int,
        obs: dict[str, Any],
        info: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        hand: str,
        instruction: str,
        gripper_closed_threshold: float,
        closed_streak: int,
        gripper_monitor: Any = None,
        model_info: Any = None,
    ) -> dict[str, Any]:
        opening, compact = self._selected_gripper_opening(obs, hand)
        total_env_steps = int(env_steps)
        if isinstance(info, dict):
            rpent = info.get("_rpent")
            if isinstance(rpent, dict) and "total_env_steps" in rpent:
                total_env_steps = int(rpent["total_env_steps"])
        return {
            "chunk": int(chunk),
            "env_steps": int(env_steps),
            "total_env_steps": total_env_steps,
            "instruction": instruction,
            "selected_hand": hand,
            "selected_gripper_opening": opening,
            "gripper_closed_threshold": float(gripper_closed_threshold),
            "gripper_closed": bool(opening <= gripper_closed_threshold),
            "consecutive_closed_chunks": int(closed_streak),
            "local_gripper_monitor": _jsonable(gripper_monitor),
            "raw_proprio": _jsonable(np.asarray(obs["states"], dtype=np.float32)),
            "policy_state": _jsonable(compact),
            "policy_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
            "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
            "reward": _jsonable(reward),
            "terminated": _as_bool(terminated),
            "truncated": _as_bool(truncated),
            "info": _jsonable(info),
            "model": _jsonable(model_info),
            "task_success": official_task_success(info),
            "official_success_source": 'info["done"]["success"]',
        }

    @staticmethod
    def _pi0_nav_pick_state_record(
        *,
        chunk: int,
        env_steps: int,
        obs: dict[str, Any],
        info: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        instruction: str,
        monitor: Any = None,
        model_info: Any = None,
    ) -> dict[str, Any]:
        compact = extract_policy_state(np.asarray(obs["states"], dtype=np.float32))
        total_env_steps = int(env_steps)
        if isinstance(monitor, dict) and "total_env_steps" in monitor:
            total_env_steps = int(monitor["total_env_steps"])
        return {
            "chunk": int(chunk),
            "env_steps": int(env_steps),
            "total_env_steps": total_env_steps,
            "instruction": instruction,
            "predicted_action_shape": (
                None if chunk == 0 else [DEFAULT_ACTION_CHUNK, 23]
            ),
            "local_grasp_success": bool(
                isinstance(monitor, dict) and monitor.get("local_grasp_success") is True
            ),
            "pi0_nav_pick_monitor": _jsonable(monitor),
            "raw_proprio": _jsonable(np.asarray(obs["states"], dtype=np.float32)),
            "policy_state": _jsonable(compact),
            "policy_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
            "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
            "reward": _jsonable(reward),
            "terminated": _as_bool(terminated),
            "truncated": _as_bool(truncated),
            "info": _jsonable(info),
            "model": _jsonable(model_info),
            "task_success": official_task_success(info),
            "official_success_source": 'info["done"]["success"]',
        }

    def pi0_navigate_to(
        self,
        instruction: str,
        max_chunks: int = 4,
    ) -> dict[str, Any]:
        """Run one bounded base-only VLA segment and pause for visual review."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pi0_navigate_to_segment_counter += 1
        segment_index = self._pi0_navigate_to_segment_counter
        segment_dir = (
            self.output_dir / "pi0_navigate_to" / f"segment_{segment_index:04d}"
        )
        states_path = segment_dir / "states.json"
        result_path = segment_dir / "result.json"
        raw_final_info_path = segment_dir / "raw_final_info.json"
        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        env_steps_used = 0
        total_env_steps = 0
        task_success = False
        terminated = False
        truncated = False
        stop_reason = "not_started"
        recoverable = True
        error: str | None = None
        last_reward: Any = None
        last_info: Any = self._current_info
        visual_artifacts: list[dict[str, Any]] = []

        try:
            if not self._hybrid_mode:
                raise RuntimeError("pi0_navigate_to is available only in hybrid mode")
            if self.env is None or self.model is None:
                raise RuntimeError("pi0_navigate_to requires both env and model")
            if self.max_episode_steps is None or self.max_episode_steps <= 0:
                raise RuntimeError(
                    "pi0_navigate_to requires a positive episode horizon"
                )
            instruction = str(instruction)
            if not instruction.strip():
                raise ValueError("instruction must be non-empty")
            if (
                isinstance(max_chunks, bool)
                or int(max_chunks) != max_chunks
                or not 1 <= int(max_chunks) <= 4
            ):
                raise ValueError("max_chunks must be an integer from 1 to 4")
            max_chunks = int(max_chunks)
            current_observation = getattr(self.env, "current_observation", None)
            if not callable(current_observation):
                raise RuntimeError(
                    "hybrid pi0_navigate_to requires env.current_observation"
                )
            obs, info = current_observation()
            self._current_observation = obs
            self._current_info = info
            last_info = info
            if isinstance(info, dict):
                rpent = info.get("_rpent")
                if isinstance(rpent, dict) and "total_env_steps" in rpent:
                    total_env_steps = int(rpent["total_env_steps"])
            if total_env_steps < 0:
                raise RuntimeError("hybrid total_env_steps must be non-negative")
            task_success = official_task_success(info)
            pi0_navigate_to_chunk_step = getattr(
                self.env,
                "pi0_navigate_to_chunk_step",
                None,
            )
            if not callable(pi0_navigate_to_chunk_step):
                raise RuntimeError(
                    "hybrid pi0_navigate_to requires env.pi0_navigate_to_chunk_step"
                )
            stop_reason = "task_success" if task_success else "running"

            while (
                not task_success
                and not terminated
                and not truncated
                and chunks_used < max_chunks
                and total_env_steps < self.max_episode_steps
            ):
                model_obs = dict(obs)
                model_obs["task_descriptions"] = instruction
                predicted, model_info = self.model.predict_action_batch(
                    model_obs,
                    mode="eval",
                )
                actions = validate_action_chunk(
                    predicted,
                    max_horizon=self.action_horizon,
                )
                remaining = self.max_episode_steps - total_env_steps
                actions = actions[: min(8, remaining)]
                if actions.shape[0] < 1:
                    stop_reason = "horizon"
                    break
                chunk_index = chunks_used + 1
                obs, reward, term, trunc, info = pi0_navigate_to_chunk_step(
                    actions,
                    segment_index=segment_index,
                    chunk_index=chunk_index,
                )
                chunks_used = chunk_index
                last_reward = reward
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)
                task_success = official_task_success(info)
                rpent_info = info.get("_rpent") if isinstance(info, dict) else None
                executed_steps = (
                    rpent_info.get("executed_steps")
                    if isinstance(rpent_info, dict)
                    else None
                )
                if (
                    isinstance(executed_steps, bool)
                    or not isinstance(executed_steps, (int, np.integer))
                    or not 1 <= int(executed_steps) <= actions.shape[0]
                ):
                    raise RuntimeError(
                        "invalid env executed_steps for pi0_navigate_to: "
                        f"{executed_steps!r}"
                    )
                reported_total = (
                    rpent_info.get("total_env_steps")
                    if isinstance(rpent_info, dict)
                    else None
                )
                if (
                    isinstance(reported_total, bool)
                    or not isinstance(reported_total, (int, np.integer))
                    or int(reported_total) < total_env_steps + int(executed_steps)
                ):
                    raise RuntimeError(
                        "invalid env total_env_steps for pi0_navigate_to: "
                        f"{reported_total!r}"
                    )
                env_steps_used += int(executed_steps)
                total_env_steps = int(reported_total)
                monitor = (
                    rpent_info.get("pi0_navigate_to_monitor")
                    if isinstance(rpent_info, dict)
                    else None
                )
                if not isinstance(monitor, dict):
                    raise RuntimeError(
                        "pi0_navigate_to env result omitted safety monitor"
                    )
                review = monitor.get("visual_review")
                if not isinstance(review, dict):
                    raise RuntimeError(
                        "pi0_navigate_to env result omitted visual artifacts"
                    )
                visual_artifacts.append(_jsonable(review))
                states.append(
                    {
                        "segment_index": segment_index,
                        "chunk_index": chunk_index,
                        "instruction": instruction,
                        "predicted_actions": int(len(predicted)),
                        "executed_action_limit": 8,
                        "executed_steps": int(executed_steps),
                        "env_steps_used": env_steps_used,
                        "total_env_steps": total_env_steps,
                        "model": _jsonable(model_info),
                        "monitor": _jsonable(monitor),
                        "task_success": task_success,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                _write_json_atomic(states_path, states)
                self._current_observation = obs
                self._current_info = info

                if task_success:
                    stop_reason = "task_success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                elif bool(monitor.get("safety_stop", False)):
                    stop_reason = str(monitor.get("stop_reason") or "safety_stop")
                    recoverable = False
                elif total_env_steps >= self.max_episode_steps:
                    stop_reason = "horizon"
                elif chunks_used >= max_chunks:
                    stop_reason = "visual_review_pause"
                else:
                    stop_reason = "running"
                if not recoverable:
                    break

            if stop_reason == "running":
                stop_reason = (
                    "horizon"
                    if total_env_steps >= self.max_episode_steps
                    else "visual_review_pause"
                )
        except Exception as exc:
            logger.exception("pi0_navigate_to failed")
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "error"
            recoverable = False

        result = {
            "_finish": bool(task_success or terminated or truncated),
            "name": "pi0_navigate_to",
            "instruction": instruction,
            "segment_index": segment_index,
            "segment_completed": stop_reason == "visual_review_pause",
            "primitive_success": False,
            "task_success": bool(task_success),
            "official_success_source": 'info["done"]["success"]',
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "recoverable": bool(recoverable),
            "suggested_next_tool": (
                None
                if task_success or terminated or truncated or not recoverable
                else "observe"
            ),
            "visual_review_required": not bool(task_success),
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "actions_per_chunk_limit": 8,
            "env_steps_used": env_steps_used,
            "total_env_steps": total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "states_path": str(states_path),
            "result_path": str(result_path),
            "raw_final_info_path": str(raw_final_info_path),
            "visual_artifacts": visual_artifacts,
            "visual_review": {
                "views": ["head", "left_wrist", "right_wrist"],
                "chunk_artifacts": visual_artifacts,
                "video_path": str(self.video_path),
            },
            "video_path": str(self.video_path),
            "video_fps": 15,
            "last_reward": _jsonable(last_reward),
            "last_info": _public_info_summary(last_info),
            "elapsed_s": round(time.time() - started, 2),
            "error": error,
        }
        _write_json_atomic(states_path, states)
        _write_json_atomic(raw_final_info_path, _jsonable(last_info))
        _write_json_atomic(result_path, result)
        if self._current_observation is not None:
            wrists = np.asarray(
                self._current_observation["wrist_images"], dtype=np.uint8
            )
            if wrists.ndim == 4 and wrists.shape[0] == 2:
                result["_image_bytes"] = _png_bytes(
                    self._current_observation["main_images"]
                )
                result["_image_cam_bytes"] = _png_bytes(wrists[0])
                result["_image_wrist_bytes"] = _png_bytes(wrists[1])
        self.last_result = result
        return result

    def pi0_nav_pick(self, *, instruction: str) -> dict[str, Any]:
        """Run continuous full Pi0 chunks until strict env-side grasp handoff."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        states_path = self.output_dir / "pi0_nav_pick_states.json"
        result_path = self.output_dir / "pi0_nav_pick_result.json"
        raw_final_info_path = self.output_dir / "pi0_nav_pick_raw_final_info.json"
        for path in (states_path, result_path, raw_final_info_path):
            if path.exists():
                path.unlink()

        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        full_chunks_executed = 0
        env_steps_used = 0
        vla_env_steps_used = 0
        handoff_env_steps_used = 0
        chunk_visual_reviews: list[dict[str, Any]] = []
        total_env_steps = 0
        task_success = False
        local_grasp_success = False
        terminated = False
        truncated = False
        stop_reason = "not_started"
        last_reward: Any = None
        last_info: Any = self._current_info
        last_monitor: dict[str, Any] | None = None
        vla_disable_confirmation: dict[str, Any] | None = None
        vla_health_after_disable: dict[str, Any] | None = None
        paused_runtime_finalization: dict[str, Any] | None = None
        vla_actions_disabled = False
        formal_handoff_ready = False
        strict_local_grasp_success = False
        usable_post_pick_saved = False
        error: str | None = None
        obs: dict[str, Any] | None = self._current_observation

        try:
            if self.env is None or self.model is None:
                raise RuntimeError("pi0_nav_pick requires env and model")
            if self.max_episode_steps is None or self.max_episode_steps <= 0:
                raise ValueError("pi0_nav_pick requires a positive episode horizon")
            if self.action_horizon != DEFAULT_ACTION_CHUNK:
                raise ValueError(
                    "pi0_nav_pick requires action_horizon=32 for complete chunks"
                )
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a non-empty string")
            pi0_nav_pick_chunk_step = getattr(self.env, "pi0_nav_pick_chunk_step", None)
            if not callable(pi0_nav_pick_chunk_step):
                raise RuntimeError("pi0_nav_pick requires env.pi0_nav_pick_chunk_step")

            if obs is None:
                obs, info = self.env.reset()
                self._current_observation = obs
                self._current_info = info
            else:
                info = self._current_info
            last_info = info
            if isinstance(info, dict):
                rpent = info.get("_rpent")
                if isinstance(rpent, dict) and "total_env_steps" in rpent:
                    total_env_steps = int(rpent["total_env_steps"])
            if total_env_steps < 0:
                raise RuntimeError(
                    "pi0_nav_pick initial total_env_steps must be non-negative"
                )
            task_success = official_task_success(info)
            states.append(
                self._pi0_nav_pick_state_record(
                    chunk=0,
                    env_steps=0,
                    obs=obs,
                    info=info,
                    reward=None,
                    terminated=False,
                    truncated=False,
                    instruction=instruction,
                )
            )
            _write_json_atomic(states_path, states)
            stop_reason = "task_success" if task_success else "running"

            while (
                stop_reason == "running"
                and not task_success
                and not local_grasp_success
                and not terminated
                and not truncated
                and total_env_steps < self.max_episode_steps
            ):
                model_obs = dict(obs)
                # Preserve the configured language byte-for-byte. Adding a
                # navigation or hand hint changes the Pi0 policy distribution.
                model_obs["task_descriptions"] = instruction
                predicted, model_info = self.model.predict_action_batch(
                    model_obs, mode="eval"
                )
                actions = validate_action_chunk(
                    predicted, max_horizon=DEFAULT_ACTION_CHUNK
                )
                if actions.shape != (DEFAULT_ACTION_CHUNK, 23):
                    raise RuntimeError(
                        "pi0_nav_pick model must return exactly one complete "
                        f"[32,23] chunk, got {actions.shape}"
                    )

                previous_total = total_env_steps
                chunk_index = chunks_used + 1
                obs, reward, term, trunc, info = pi0_nav_pick_chunk_step(
                    actions,
                    chunk_index=chunk_index,
                )
                chunks_used = chunk_index
                last_reward = reward
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)
                task_success = official_task_success(info)

                rpent = info.get("_rpent") if isinstance(info, dict) else None
                monitor = (
                    rpent.get("pi0_nav_pick_monitor")
                    if isinstance(rpent, dict)
                    else None
                )
                if not isinstance(monitor, dict):
                    raise RuntimeError(
                        "pi0_nav_pick env result omitted pi0_nav_pick_monitor"
                    )
                missing = sorted(set(_PI0_NAV_PICK_MONITOR_FIELDS) - set(monitor))
                if missing:
                    raise RuntimeError(
                        f"pi0_nav_pick monitor missing required fields: {missing}"
                    )
                monitor = {
                    field: monitor[field]
                    for field in (
                        *_PI0_NAV_PICK_MONITOR_FIELDS,
                        *_PI0_NAV_PICK_OPTIONAL_MONITOR_FIELDS,
                    )
                    if field in monitor
                }
                visual_review = monitor.get("visual_review")
                if visual_review is not None:
                    if not isinstance(visual_review, dict):
                        raise RuntimeError(
                            "pi0_nav_pick visual_review must be a mapping"
                        )
                    views = visual_review.get("views")
                    metadata_path = visual_review.get("metadata_path")
                    if (
                        not isinstance(visual_review.get("capture_group_id"), str)
                        or not visual_review["capture_group_id"]
                        or not isinstance(metadata_path, str)
                        or not metadata_path
                        or not isinstance(views, dict)
                        or set(views) != {"head", "left_wrist", "right_wrist"}
                    ):
                        raise RuntimeError("pi0_nav_pick visual_review is incomplete")
                    review_paths = [metadata_path]
                    for camera, view in views.items():
                        if (
                            not isinstance(view, dict)
                            or not isinstance(view.get("path"), str)
                            or not view["path"]
                            or view.get("frame_id") is None
                        ):
                            raise RuntimeError(
                                "pi0_nav_pick visual_review has invalid "
                                f"{camera} evidence"
                            )
                        review_paths.append(view["path"])
                    missing_review_paths = [
                        path for path in review_paths if not Path(path).is_file()
                    ]
                    if missing_review_paths:
                        raise RuntimeError(
                            "pi0_nav_pick visual_review artifacts are missing: "
                            f"{missing_review_paths}"
                        )
                    chunk_visual_reviews.append(_jsonable(visual_review))
                monitor_stop_reason = monitor.get("stop_reason")
                if monitor_stop_reason is not None and (
                    not isinstance(monitor_stop_reason, str) or not monitor_stop_reason
                ):
                    raise RuntimeError(
                        "pi0_nav_pick monitor stop_reason must be null or non-empty"
                    )
                # Persist only the public monitor allowlist. Simulator-private
                # debugging fields must not cross into agent artifacts.
                public_info = dict(info)
                public_rpent = dict(rpent)
                public_rpent["pi0_nav_pick_monitor"] = monitor
                public_info["_rpent"] = public_rpent
                info = public_info
                last_info = info
                executed_steps = monitor["executed_steps"]
                if (
                    isinstance(executed_steps, bool)
                    or not isinstance(executed_steps, (int, np.integer))
                    or not 1 <= int(executed_steps) <= DEFAULT_ACTION_CHUNK
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick executed_steps: {executed_steps!r}"
                    )
                handoff_env_steps = monitor["handoff_env_steps"]
                if (
                    isinstance(handoff_env_steps, bool)
                    or not isinstance(handoff_env_steps, (int, np.integer))
                    or int(handoff_env_steps) < 0
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick handoff_env_steps: {handoff_env_steps!r}"
                    )
                reported_total = monitor["total_env_steps"]
                if (
                    isinstance(reported_total, bool)
                    or not isinstance(reported_total, (int, np.integer))
                    or int(reported_total) < previous_total + int(executed_steps)
                    or int(reported_total) > self.max_episode_steps
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick total_env_steps: {reported_total!r}"
                    )
                if not isinstance(monitor["local_grasp_success"], bool):
                    raise RuntimeError(
                        "pi0_nav_pick local_grasp_success must be boolean"
                    )
                local_grasp_success = monitor["local_grasp_success"] is True
                usable_post_pick_saved = bool(
                    monitor.get("usable_post_pick_saved", False)
                )
                strict_local_grasp_success = bool(
                    monitor.get("strict_local_grasp_success", local_grasp_success)
                )
                if "warnings" in monitor and not isinstance(
                    monitor.get("warnings"), list
                ):
                    raise RuntimeError("pi0_nav_pick warnings must be a list")
                failed_local_handoff = bool(
                    local_grasp_success
                    and monitor.get("handoff_state") == "FAILED"
                    and monitor_stop_reason is not None
                )
                derived_handoff_steps = (
                    int(reported_total) - previous_total - int(executed_steps)
                )
                if int(handoff_env_steps) != derived_handoff_steps:
                    # Preserve a strict local-grasp result even when a failed
                    # handoff exception under-reports the reload/hold steps.
                    # The monotonic env total is authoritative for accounting;
                    # this reconciliation is allowed only for an explicit
                    # local-success FAILED monitor and never promotes handoff.
                    if not failed_local_handoff:
                        raise RuntimeError(
                            f"invalid pi0_nav_pick total_env_steps: {reported_total!r}"
                        )
                    handoff_env_steps = derived_handoff_steps
                    monitor["handoff_env_steps"] = derived_handoff_steps
                if isinstance(rpent, dict):
                    for field, expected in (
                        ("executed_steps", int(executed_steps)),
                        ("handoff_env_steps", int(handoff_env_steps)),
                        ("total_env_steps", int(reported_total)),
                    ):
                        value = rpent.get(field, expected)
                        if value != expected:
                            if failed_local_handoff and field == "handoff_env_steps":
                                continue
                            raise RuntimeError(
                                "pi0_nav_pick monitor/accounting mismatch: "
                                f"{field}={value!r} monitor={expected!r}"
                            )
                # Keep persisted/public accounting internally consistent after
                # the narrow FAILED reconciliation above.
                public_rpent["executed_steps"] = int(executed_steps)
                public_rpent["handoff_env_steps"] = int(handoff_env_steps)
                public_rpent["total_env_steps"] = int(reported_total)
                public_rpent["pi0_nav_pick_monitor"] = monitor
                public_info["_rpent"] = public_rpent
                info = public_info
                last_info = info
                if not isinstance(monitor["per_hand"], dict) or not isinstance(
                    monitor["current_criteria"], dict
                ):
                    raise RuntimeError("pi0_nav_pick monitor criteria must be mappings")
                handoff_state = monitor.get("handoff_state")
                expected_vla_enabled = handoff_state not in {
                    "FAILED",
                    "OFFICIAL_SUCCESS",
                }
                if monitor["vla_actions_enabled"] is not expected_vla_enabled:
                    raise RuntimeError(
                        "pi0_nav_pick VLA gate/state mismatch: "
                        f"handoff_state={handoff_state!r} "
                        f"actions_enabled={monitor['vla_actions_enabled']!r}"
                    )
                action_source = monitor.get("action_source")
                if local_grasp_success and handoff_state not in {"PAUSED", "FAILED"}:
                    raise RuntimeError(
                        "successful pi0_nav_pick requires handoff_state='PAUSED' "
                        "or an explicit FAILED stop"
                    )
                if (
                    local_grasp_success
                    and handoff_state == "FAILED"
                    and not (monitor_stop_reason)
                ):
                    raise RuntimeError(
                        "FAILED local pi0_nav_pick handoff requires stop_reason"
                    )
                if (
                    local_grasp_success
                    and handoff_state == "PAUSED"
                    and usable_post_pick_saved
                ):
                    expected_action_source = action_source
                    if action_source not in {"curobo", "pi0_vla"}:
                        raise RuntimeError(
                            "saved post-pick runtime has invalid action_source"
                        )
                elif local_grasp_success and handoff_state == "PAUSED":
                    expected_action_source = "curobo"
                else:
                    expected_action_source = "pi0_vla"
                if action_source != expected_action_source:
                    raise RuntimeError(
                        "pi0_nav_pick action_source mismatch: "
                        f"expected={expected_action_source!r} "
                        f"actual={action_source!r}"
                    )
                checkpoint_fields = (
                    "state_checkpoint_path",
                    "paused_runtime_path",
                )
                if (
                    int(executed_steps) != DEFAULT_ACTION_CHUNK
                    and not local_grasp_success
                    and not task_success
                    and not terminated
                    and not truncated
                    and int(reported_total) != self.max_episode_steps
                    and monitor_stop_reason is None
                ):
                    raise RuntimeError(
                        "pi0_nav_pick env truncated a chunk without a terminal "
                        "or local-success reason"
                    )

                # Account for the simulator return before validating the
                # post-pick artifacts. If handoff finalization fails, the
                # result must still preserve the actual executed steps and
                # strict local validator evidence.
                env_steps_used += int(executed_steps)
                env_steps_used += int(handoff_env_steps)
                vla_env_steps_used += int(executed_steps)
                handoff_env_steps_used += int(handoff_env_steps)
                total_env_steps = int(reported_total)
                if int(executed_steps) == DEFAULT_ACTION_CHUNK:
                    full_chunks_executed += 1
                last_monitor = dict(monitor)
                states.append(
                    self._pi0_nav_pick_state_record(
                        chunk=chunks_used,
                        env_steps=env_steps_used,
                        obs=obs,
                        info=info,
                        reward=reward,
                        terminated=term,
                        truncated=trunc,
                        instruction=instruction,
                        monitor=monitor,
                        model_info=model_info,
                    )
                )
                self._current_observation = obs
                self._current_info = info

                if local_grasp_success and monitor["handoff_state"] == "PAUSED":
                    if monitor["held_hand"] not in {"left", "right"}:
                        raise RuntimeError(
                            "successful pi0_nav_pick monitor lacks held_hand"
                        )
                    missing_paths = [
                        field
                        for field in (
                            "validator_trace_path",
                            *checkpoint_fields,
                        )
                        if not isinstance(monitor[field], str) or not monitor[field]
                    ]
                    if missing_paths:
                        raise RuntimeError(
                            "successful pi0_nav_pick handoff lacks paths: "
                            f"{missing_paths}"
                        )
                    if monitor["handoff_state"] != "PAUSED":
                        raise RuntimeError(
                            "successful pi0_nav_pick requires handoff_state='PAUSED'"
                        )
                    missing_artifacts = [
                        field
                        for field in (
                            "validator_trace_path",
                            *checkpoint_fields,
                        )
                        if not Path(str(monitor[field])).is_file()
                    ]
                    if missing_artifacts:
                        raise RuntimeError(
                            "successful pi0_nav_pick handoff artifacts are missing: "
                            f"{missing_artifacts}"
                        )
                    # A debug-save PAUSED artifact may truthfully retain
                    # action_source=pi0_vla when controller handoff failed.
                    # Preserve it and disable VLA, but do not promote it to a
                    # successful CuRobo handoff.
                    formal_handoff_ready = bool(action_source == "curobo")

                    disable_actions = getattr(self.model, "disable_actions", None)
                    healthz = getattr(self.model, "healthz", None)
                    if not callable(disable_actions) or not callable(healthz):
                        raise RuntimeError(
                            "pi0_nav_pick VLA client lacks disable_actions/healthz"
                        )
                    disabled = disable_actions()
                    if (
                        not isinstance(disabled, dict)
                        or disabled.get("actions_enabled") is not False
                    ):
                        raise RuntimeError("pi0_nav_pick VLA disable was not confirmed")
                    health = healthz()
                    if (
                        not isinstance(health, dict)
                        or health.get("actions_enabled") is not False
                    ):
                        raise RuntimeError(
                            "pi0_nav_pick VLA health still permits actions"
                        )
                    disabled_pid = disabled.get("pid")
                    health_pid = health.get("pid")
                    if (
                        isinstance(disabled_pid, bool)
                        or not isinstance(disabled_pid, (int, np.integer))
                        or int(disabled_pid) <= 0
                        or isinstance(health_pid, bool)
                        or not isinstance(health_pid, (int, np.integer))
                        or int(health_pid) != int(disabled_pid)
                    ):
                        raise RuntimeError(
                            "pi0_nav_pick VLA disable/health process identity "
                            f"mismatch: disable_pid={disabled_pid!r} "
                            f"health_pid={health_pid!r}"
                        )
                    vla_endpoint = getattr(self.model, "endpoint", None)
                    if not isinstance(vla_endpoint, str) or not vla_endpoint:
                        raise RuntimeError(
                            "pi0_nav_pick VLA client endpoint is unavailable"
                        )
                    vla_disable_confirmation = dict(disabled)
                    vla_health_after_disable = dict(health)
                    vla_actions_disabled = True

                    vla_status = {
                        "actions_enabled": False,
                        "endpoint": vla_endpoint,
                        "pid": int(disabled_pid),
                        "disable_confirmation": _jsonable(disabled),
                        "healthz": _jsonable(health),
                    }
                    finalize_paused_runtime = getattr(
                        self.env, "finalize_paused_runtime", None
                    )
                    if callable(finalize_paused_runtime):
                        finalized = finalize_paused_runtime(vla_status)
                        if (
                            not isinstance(finalized, dict)
                            or finalized.get("vla_actions_enabled") is not False
                        ):
                            raise RuntimeError(
                                "env did not confirm paused runtime finalization"
                            )
                        paused_runtime_finalization = dict(finalized)
                    else:
                        paused_runtime_path = Path(monitor["paused_runtime_path"])
                        try:
                            paused_runtime = json.loads(
                                paused_runtime_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError) as exc:
                            raise RuntimeError(
                                "pi0_nav_pick paused runtime artifact is invalid: "
                                f"{exc}"
                            ) from exc
                        if not isinstance(paused_runtime, dict):
                            raise RuntimeError(
                                "pi0_nav_pick paused runtime must be a JSON object"
                            )
                        paused_runtime.update(
                            {
                                "vla_actions_enabled": False,
                                "vla_action_gate_confirmed": True,
                                "vla_endpoint": vla_endpoint,
                                "vla_pid": int(disabled_pid),
                                "vla_disable_confirmation": _jsonable(disabled),
                                "vla_health_after_disable": _jsonable(health),
                            }
                        )
                        _write_json_atomic(paused_runtime_path, paused_runtime)
                        paused_runtime_finalization = {
                            "paused_runtime_path": str(paused_runtime_path),
                            "vla_actions_enabled": False,
                            "vla_endpoint": vla_endpoint,
                            "vla_pid": int(disabled_pid),
                            "source": "agent_atomic_fallback",
                        }
                elif local_grasp_success:
                    if monitor["held_hand"] not in {"left", "right"}:
                        raise RuntimeError(
                            "successful pi0_nav_pick monitor lacks held_hand"
                        )
                    # An explicit FAILED handoff is local evidence only. Keep it
                    # in the result, but never disable VLA here or promote it to
                    # a PAUSED/CuRobo handoff.
                    formal_handoff_ready = False
                else:
                    unexpected_paths = [
                        field
                        for field in checkpoint_fields
                        if monitor[field] is not None
                    ]
                    if unexpected_paths:
                        raise RuntimeError(
                            "unsuccessful pi0_nav_pick exposed handoff paths: "
                            f"{unexpected_paths}"
                        )

                if task_success:
                    stop_reason = "task_success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                elif formal_handoff_ready:
                    stop_reason = "local_grasp_success_handoff"
                elif usable_post_pick_saved:
                    stop_reason = "usable_post_pick_saved_handoff_not_ready"
                elif monitor_stop_reason is not None:
                    stop_reason = monitor_stop_reason
                elif local_grasp_success:
                    stop_reason = "handoff_failed"
                elif total_env_steps >= self.max_episode_steps:
                    stop_reason = "horizon"
                else:
                    stop_reason = "running"
                _write_json_atomic(states_path, states)

            if stop_reason == "running":
                stop_reason = "horizon"
        except Exception as exc:
            logger.exception("pi0_nav_pick failed")
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "error"

        monitor_public = _jsonable(last_monitor)
        primitive_success = bool(
            formal_handoff_ready and vla_actions_disabled and error is None
        )
        result = {
            "_finish": bool(
                task_success or terminated or truncated or not primitive_success
            ),
            "name": "pi0_nav_pick",
            "instruction": instruction,
            "primitive_success": primitive_success,
            "local_grasp_success": bool(local_grasp_success),
            "strict_local_grasp_success": bool(strict_local_grasp_success),
            "usable_post_pick_saved": bool(usable_post_pick_saved),
            "save_policy": (
                last_monitor.get("save_policy")
                if isinstance(last_monitor, dict)
                else None
            ),
            "warnings": (
                _jsonable(last_monitor.get("warnings", []))
                if isinstance(last_monitor, dict)
                else []
            ),
            "local_success_source": (
                'info["_rpent"]["pi0_nav_pick_monitor"]["local_grasp_success"]'
            ),
            "task_success": bool(task_success),
            "official_success_source": 'info["done"]["success"]',
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "chunks_used": chunks_used,
            "full_chunks_executed": full_chunks_executed,
            "env_steps_used": env_steps_used,
            "vla_env_steps_used": vla_env_steps_used,
            "handoff_env_steps_used": handoff_env_steps_used,
            "total_env_steps": total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "required_action_shape": [DEFAULT_ACTION_CHUNK, 23],
            "held_hand": (
                last_monitor.get("held_hand")
                if isinstance(last_monitor, dict)
                else None
            ),
            "per_hand": (
                _jsonable(last_monitor.get("per_hand"))
                if isinstance(last_monitor, dict)
                else None
            ),
            "current_criteria": (
                _jsonable(last_monitor.get("current_criteria"))
                if isinstance(last_monitor, dict)
                else None
            ),
            "validator_trace_path": (
                last_monitor.get("validator_trace_path")
                if isinstance(last_monitor, dict)
                else None
            ),
            "state_checkpoint_path": (
                last_monitor.get("state_checkpoint_path")
                if isinstance(last_monitor, dict)
                else None
            ),
            "handoff_state": (
                last_monitor.get("handoff_state")
                if isinstance(last_monitor, dict)
                else None
            ),
            "action_source": (
                last_monitor.get("action_source")
                if isinstance(last_monitor, dict)
                else None
            ),
            "vla_actions_enabled": (
                False
                if vla_actions_disabled
                else last_monitor.get("vla_actions_enabled")
                if isinstance(last_monitor, dict)
                else None
            ),
            "vla_actions_enabled_before_handoff": (
                last_monitor.get("vla_actions_enabled")
                if isinstance(last_monitor, dict)
                else None
            ),
            "vla_actions_disabled": bool(vla_actions_disabled),
            "vla_disable_confirmation": _jsonable(vla_disable_confirmation),
            "vla_health_after_disable": _jsonable(vla_health_after_disable),
            "vla_endpoint": (
                getattr(self.model, "endpoint", None) if vla_actions_disabled else None
            ),
            "vla_pid": (
                int(vla_disable_confirmation["pid"])
                if isinstance(vla_disable_confirmation, dict)
                and isinstance(vla_disable_confirmation.get("pid"), (int, np.integer))
                and not isinstance(vla_disable_confirmation.get("pid"), bool)
                else None
            ),
            "paused_runtime_finalization": _jsonable(paused_runtime_finalization),
            "paused_runtime_path": (
                last_monitor.get("paused_runtime_path")
                if isinstance(last_monitor, dict)
                else None
            ),
            "last_pi0_nav_pick_monitor": monitor_public,
            "elapsed_s": round(time.time() - started, 2),
            "states_path": str(states_path),
            "result_path": str(result_path),
            "raw_final_info_path": str(raw_final_info_path),
            "action_trace_path": str(self.output_dir / "behavior_action_trace.jsonl"),
            "manifest_path": str(self.output_dir / MANIFEST_FILENAME),
            "video_path": str(self.video_path),
            "video_fps": 15,
            "visual_review_required": True,
            "visual_review": {
                "views": ["head", "left_wrist", "right_wrist"],
                "video_path": str(self.video_path),
                "chunk_artifacts": chunk_visual_reviews,
                "validator_trace_path": (
                    last_monitor.get("validator_trace_path")
                    if isinstance(last_monitor, dict)
                    else None
                ),
            },
            "last_reward": _jsonable(last_reward),
            "last_info": _public_info_summary(last_info),
            "error": error,
        }
        _write_json_atomic(states_path, states)
        _write_json_atomic(raw_final_info_path, _jsonable(last_info))
        _write_json_atomic(result_path, result)
        if self._current_observation is not None:
            wrists = np.asarray(
                self._current_observation["wrist_images"], dtype=np.uint8
            )
            if wrists.ndim == 4 and wrists.shape[0] == 2:
                result["_image_bytes"] = _png_bytes(
                    self._current_observation["main_images"]
                )
                result["_image_cam_bytes"] = _png_bytes(wrists[0])
                result["_image_wrist_bytes"] = _png_bytes(wrists[1])
        self.last_result = result
        return result

    def pi0_pick(
        self,
        hand: str,
        instruction: str,
        max_chunks: int = 24,
        gripper_closed_threshold: float = 0.045,
        required_closed_chunks: int = 1,
        stop_on_closure_candidate: bool = False,
        post_candidate_chunks: int = 4,
    ) -> dict[str, Any]:
        """Run a bounded local VLA grasp loop without claiming grasp from closure.

        Selected-hand compact gripper closure is only an early-stop candidate.
        A caller-supplied public-observation validator is required before local
        or primitive success can become true, and visual review remains required
        even when that validator accepts the candidate.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        states_path = self.output_dir / "pi0_pick_states.json"
        result_path = self.output_dir / "pi0_pick_result.json"
        raw_final_info_path = self.output_dir / "pi0_pick_raw_final_info.json"
        for path in (states_path, result_path, raw_final_info_path):
            if path.exists():
                path.unlink()

        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        env_steps_used = 0
        task_success = False
        local_grasp_success = False
        local_gripper_closure_detected = False
        closure_candidate_paused = False
        first_candidate_env_step: int | None = None
        first_candidate_chunk: int | None = None
        post_candidate_chunks_used = 0
        terminated = False
        truncated = False
        closed_streak = 0
        stop_reason = "not_started"
        last_reward: Any = None
        last_info: Any = self._current_info
        last_gripper_opening: float | None = None
        validator_result: bool | None = None
        last_gripper_monitor: dict[str, Any] | None = None
        error: str | None = None
        model_instruction: str | None = None
        total_env_steps = 0

        try:
            if self.env is None or self.model is None:
                raise RuntimeError("pi0_pick requires env and model")
            if self.max_episode_steps is None or self.max_episode_steps <= 0:
                raise ValueError("pi0_pick requires positive max_episode_steps")
            if self.action_horizon <= 0:
                raise ValueError("action_horizon must be positive")
            if hand not in {"left", "right"}:
                raise ValueError("hand must be 'left' or 'right'")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a non-empty string")
            instruction = instruction.strip()
            # Preserve a caller-supplied policy instruction verbatim. Pi0.5 is
            # trained on natural task language, and injecting a second policy
            # sentence can materially change its action distribution. The hand
            # parameter selects the measured gripper and local validator; the
            # standard CLI instruction names that hand explicitly.
            model_instruction = instruction.replace("selected hand", f"{hand} hand")
            if isinstance(max_chunks, bool) or int(max_chunks) != max_chunks:
                raise ValueError("max_chunks must be a positive integer")
            max_chunks = int(max_chunks)
            if max_chunks <= 0:
                raise ValueError("max_chunks must be a positive integer")
            threshold = float(gripper_closed_threshold)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    "gripper_closed_threshold must be finite and non-negative"
                )
            if (
                isinstance(required_closed_chunks, bool)
                or int(required_closed_chunks) != required_closed_chunks
            ):
                raise ValueError("required_closed_chunks must be a positive integer")
            required_closed_chunks = int(required_closed_chunks)
            if required_closed_chunks <= 0:
                raise ValueError("required_closed_chunks must be a positive integer")
            if not isinstance(stop_on_closure_candidate, bool):
                raise ValueError("stop_on_closure_candidate must be a boolean")
            if self._hybrid_mode:
                # Hybrid always yields control back to the visual planner after
                # the bounded post-candidate motion, even if an LLM omits the
                # optional schema field.
                stop_on_closure_candidate = True
            if (
                isinstance(post_candidate_chunks, bool)
                or int(post_candidate_chunks) != post_candidate_chunks
                or int(post_candidate_chunks) < 0
            ):
                raise ValueError("post_candidate_chunks must be a non-negative integer")
            post_candidate_chunks = int(post_candidate_chunks)

            if self._hybrid_mode:
                current_observation = getattr(self.env, "current_observation", None)
                if not callable(current_observation):
                    raise RuntimeError(
                        "hybrid pi0_pick requires env.current_observation"
                    )
                # Planner motion may have changed all cameras and proprio. Refresh
                # the current episode in place; a reset here would silently start
                # a different evaluation episode.
                obs, info = current_observation()
                self._current_observation = obs
                self._current_info = info
            elif self._current_observation is None:
                obs, info = self.env.reset()
                self._current_observation = obs
                self._current_info = info
            else:
                obs = self._current_observation
                info = self._current_info
            last_info = info
            if isinstance(info, dict):
                rpent = info.get("_rpent")
                if isinstance(rpent, dict) and "total_env_steps" in rpent:
                    total_env_steps = int(rpent["total_env_steps"])
            if self._hybrid_mode and total_env_steps < 0:
                raise RuntimeError("hybrid total_env_steps must be non-negative")
            initial_state = self._pi0_state_record(
                chunk=0,
                env_steps=0,
                obs=obs,
                info=info,
                reward=None,
                terminated=False,
                truncated=False,
                hand=hand,
                instruction=instruction,
                gripper_closed_threshold=threshold,
                # The initial state never contributes to a post-action closure
                # streak, even when the restored gripper starts closed.
                closed_streak=0,
            )
            states.append(initial_state)
            last_gripper_opening = initial_state["selected_gripper_opening"]
            task_success = initial_state["task_success"]
            stop_reason = "task_success" if task_success else "running"
            _write_json_atomic(states_path, states)

            while (
                not task_success
                and not local_grasp_success
                and not terminated
                and not truncated
                and chunks_used < max_chunks
                and not closure_candidate_paused
                and total_env_steps < self.max_episode_steps
            ):
                model_obs = dict(obs)
                model_obs["task_descriptions"] = model_instruction
                actions, model_info = self.model.predict_action_batch(
                    model_obs, mode="eval"
                )
                actions = validate_action_chunk(
                    actions,
                    max_horizon=self.action_horizon,
                )
                remaining = self.max_episode_steps - total_env_steps
                actions = actions[:remaining]
                pi0_chunk_step = getattr(self.env, "pi0_chunk_step", None)
                if callable(pi0_chunk_step):
                    obs, reward, term, trunc, info = pi0_chunk_step(
                        actions,
                        hand=hand,
                        gripper_closed_threshold=threshold,
                        required_closed_steps=3,
                        stop_on_candidate=(
                            self._local_grasp_validator is not None
                            and not stop_on_closure_candidate
                        ),
                    )
                else:
                    # Direct unit fakes written before the isolated PI0 RPC may
                    # still implement the common env protocol. Production
                    # BehaviorEnvClient always uses pi0_chunk_step in this mode.
                    obs, reward, term, trunc, info = self.env.chunk_step(actions)

                chunks_used += 1
                executed_steps: Any = actions.shape[0]
                if isinstance(info, dict):
                    executed_steps = info.get("_rpent", {}).get(
                        "executed_steps", executed_steps
                    )
                if (
                    isinstance(executed_steps, bool)
                    or int(executed_steps) != executed_steps
                    or not 1 <= int(executed_steps) <= actions.shape[0]
                ):
                    raise RuntimeError(
                        f"invalid env executed_steps for pi0_pick: {executed_steps!r}"
                    )
                env_steps_used += int(executed_steps)
                reported_total_env_steps: Any = None
                if isinstance(info, dict):
                    rpent_info = info.get("_rpent")
                    if isinstance(rpent_info, dict):
                        reported_total_env_steps = rpent_info.get("total_env_steps")
                if reported_total_env_steps is None:
                    total_env_steps += int(executed_steps)
                elif (
                    isinstance(reported_total_env_steps, bool)
                    or int(reported_total_env_steps) != reported_total_env_steps
                    or int(reported_total_env_steps)
                    < total_env_steps + int(executed_steps)
                ):
                    raise RuntimeError(
                        "invalid env total_env_steps for pi0_pick: "
                        f"{reported_total_env_steps!r}"
                    )
                else:
                    total_env_steps = int(reported_total_env_steps)
                last_reward = reward
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)
                opening, _compact = self._selected_gripper_opening(obs, hand)
                last_gripper_opening = opening
                monitor: dict[str, Any] | None = None
                if isinstance(info, dict):
                    rpent_info = info.get("_rpent")
                    if isinstance(rpent_info, dict):
                        candidate_monitor = rpent_info.get("local_gripper_monitor")
                        if isinstance(candidate_monitor, dict):
                            monitor = candidate_monitor
                last_gripper_monitor = monitor
                monitored_candidate = bool(
                    monitor is not None and monitor.get("candidate", False)
                )
                if monitored_candidate or opening <= threshold:
                    closed_streak += 1
                else:
                    closed_streak = 0
                closure_candidate = closed_streak >= required_closed_chunks
                task_success = official_task_success(info)
                if closure_candidate and first_candidate_chunk is None:
                    first_candidate_chunk = chunks_used
                    candidate_step = (
                        monitor.get("candidate_env_step")
                        if isinstance(monitor, dict)
                        else None
                    )
                    first_candidate_env_step = (
                        int(candidate_step)
                        if isinstance(candidate_step, (int, np.integer))
                        and not isinstance(candidate_step, bool)
                        else int(total_env_steps)
                    )
                elif first_candidate_chunk is not None:
                    post_candidate_chunks_used += 1

                state = self._pi0_state_record(
                    chunk=chunks_used,
                    env_steps=env_steps_used,
                    obs=obs,
                    info=info,
                    reward=reward,
                    terminated=term,
                    truncated=trunc,
                    hand=hand,
                    instruction=instruction,
                    gripper_closed_threshold=threshold,
                    closed_streak=closed_streak,
                    gripper_monitor=monitor,
                    model_info=model_info,
                )
                states.append(state)
                self._current_observation = obs
                self._current_info = info

                # Environment stop flags are never evidence of a local grasp.
                if (
                    closure_candidate
                    and not task_success
                    and not terminated
                    and not truncated
                ):
                    local_gripper_closure_detected = True
                    if (
                        self._local_grasp_validator is not None
                        and not stop_on_closure_candidate
                    ):
                        context = {
                            "hand": hand,
                            "instruction": instruction,
                            "chunk": chunks_used,
                            "env_steps": env_steps_used,
                            "selected_gripper_opening": opening,
                            "gripper_closed_threshold": threshold,
                            "consecutive_closed_chunks": closed_streak,
                            "required_closed_chunks": required_closed_chunks,
                            "local_gripper_monitor": _jsonable(monitor),
                        }
                        validator_result = bool(
                            self._local_grasp_validator(
                                _public_validator_observation(obs), context
                            )
                        )
                        local_grasp_success = validator_result

                if (
                    stop_on_closure_candidate
                    and first_candidate_chunk is not None
                    and post_candidate_chunks_used >= post_candidate_chunks
                    and not task_success
                    and not terminated
                    and not truncated
                ):
                    closure_candidate_paused = True

                if task_success:
                    stop_reason = "task_success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                elif local_grasp_success:
                    stop_reason = "local_grasp_success"
                elif closure_candidate_paused:
                    stop_reason = "closure_candidate_visual_review"
                elif total_env_steps >= self.max_episode_steps:
                    stop_reason = "horizon"
                elif chunks_used >= max_chunks:
                    stop_reason = "chunk_limit"
                else:
                    stop_reason = "running"
                _write_json_atomic(states_path, states)

            if stop_reason == "running":
                stop_reason = (
                    "horizon"
                    if total_env_steps >= self.max_episode_steps
                    else "chunk_limit"
                )
        except Exception as exc:
            logger.exception("pi0_pick failed")
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "error"

        result = {
            "_finish": (
                bool(task_success or terminated or truncated)
                if self._hybrid_mode
                else True
            ),
            "name": "pi0_pick",
            "hand": hand,
            "instruction": instruction,
            "model_instruction": model_instruction,
            "primitive_success": bool(local_grasp_success),
            "local_grasp_success": bool(local_grasp_success),
            "local_gripper_closure_detected": bool(local_gripper_closure_detected),
            "closure_candidate_paused": bool(closure_candidate_paused),
            "first_candidate_env_step": first_candidate_env_step,
            "post_candidate_chunks": post_candidate_chunks,
            "post_candidate_chunks_used": post_candidate_chunks_used,
            "stop_on_closure_candidate": stop_on_closure_candidate,
            "visual_verification_required": True,
            "visual_review_required": True,
            "local_grasp_validator_configured": (
                self._local_grasp_validator is not None
            ),
            "local_grasp_validator_result": validator_result,
            "task_success": bool(task_success),
            "official_success_source": 'info["done"]["success"]',
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "env_steps_used": env_steps_used,
            "total_env_steps": total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "gripper_closed_threshold": gripper_closed_threshold,
            "required_closed_chunks": required_closed_chunks,
            "consecutive_closed_chunks": closed_streak,
            "final_selected_gripper_opening": last_gripper_opening,
            "last_local_gripper_monitor": _jsonable(last_gripper_monitor),
            "elapsed_s": round(time.time() - started, 2),
            "states_path": str(states_path),
            "result_path": str(result_path),
            "action_trace_path": str(self.output_dir / "behavior_action_trace.jsonl"),
            "manifest_path": str(self.output_dir / MANIFEST_FILENAME),
            "video_path": str(self.video_path),
            "video_fps": 15,
            "visual_review": (
                {
                    "views": ["head", "left_wrist", "right_wrist"],
                    "video_path": str(self.video_path),
                    "result_path": str(result_path),
                }
                if self._hybrid_mode
                else {"video_path": str(self.video_path)}
            ),
            "last_reward": _jsonable(last_reward),
            "last_info": _public_info_summary(last_info),
            "raw_final_info_path": str(raw_final_info_path),
            "error": error,
        }
        _write_json_atomic(states_path, states)
        _write_json_atomic(raw_final_info_path, _jsonable(last_info))
        _write_json_atomic(result_path, result)
        if self._hybrid_mode and self._current_observation is not None:
            wrists = np.asarray(
                self._current_observation["wrist_images"], dtype=np.uint8
            )
            if wrists.ndim == 4 and wrists.shape[0] == 2:
                result["_image_bytes"] = _png_bytes(
                    self._current_observation["main_images"]
                )
                result["_image_cam_bytes"] = _png_bytes(wrists[0])
                result["_image_wrist_bytes"] = _png_bytes(wrists[1])
        self.last_result = result
        return result


__all__ = ["BehaviorPrimitives", "official_task_success"]
