"""Closed BEHAVIOR primitives for full-task, planner, and local VLA modes."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.run_manifest import MANIFEST_FILENAME
from robots.behavior.schemas import (
    DEFAULT_ACTION_CHUNK,
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    extract_policy_state,
    segment_ranges,
    validate_action_chunk,
)
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("behavior")


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


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


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
        for name in ("states.json", "raw_final_info.json", "final_result.json"):
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
            "action_trace_path": str(
                self.output_dir / "behavior_action_trace.jsonl"
            ),
            "manifest_path": str(self.output_dir / MANIFEST_FILENAME),
            "video_path": str(self.video_path),
            "video_fps": 15,
            "video_sample_every_env_steps": 4,
            "last_reward": _jsonable(last_reward),
            "last_info": _jsonable(last_info),
            "error": error,
        }
        _write_json_atomic(self.output_dir / "final_result.json", result)
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
        return self._planner_method("observe")(camera=camera)

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
        return self._planner_method("pixel_to_world")(
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
        model_info: Any = None,
    ) -> dict[str, Any]:
        opening, compact = self._selected_gripper_opening(obs, hand)
        return {
            "chunk": int(chunk),
            "env_steps": int(env_steps),
            "instruction": instruction,
            "selected_hand": hand,
            "selected_gripper_opening": opening,
            "gripper_closed_threshold": float(gripper_closed_threshold),
            "gripper_closed": bool(opening <= gripper_closed_threshold),
            "consecutive_closed_chunks": int(closed_streak),
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

    def pi0_pick(
        self,
        hand: str,
        instruction: str,
        max_chunks: int = 24,
        gripper_closed_threshold: float = 0.045,
        required_closed_chunks: int = 1,
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
        for path in (states_path, result_path):
            if path.exists():
                path.unlink()

        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        env_steps_used = 0
        task_success = False
        local_grasp_success = False
        local_gripper_closure_detected = False
        terminated = False
        truncated = False
        closed_streak = 0
        stop_reason = "not_started"
        last_reward: Any = None
        last_info: Any = self._current_info
        last_gripper_opening: float | None = None
        validator_result: bool | None = None
        error: str | None = None
        model_instruction: str | None = None

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
            model_instruction = (
                f"Use only the {hand} hand for this local grasp. {instruction}"
            )
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

            if self._current_observation is None:
                obs, info = self.env.reset()
                self._current_observation = obs
                self._current_info = info
            else:
                obs = self._current_observation
                info = self._current_info
            last_info = info
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
                and not (
                    local_gripper_closure_detected
                    and self._local_grasp_validator is None
                )
                and not terminated
                and not truncated
                and chunks_used < max_chunks
                and env_steps_used < self.max_episode_steps
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
                remaining = self.max_episode_steps - env_steps_used
                actions = actions[:remaining]
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
                    or not 0 <= int(executed_steps) <= actions.shape[0]
                ):
                    raise RuntimeError(
                        f"invalid env executed_steps for pi0_pick: {executed_steps!r}"
                    )
                env_steps_used += int(executed_steps)
                last_reward = reward
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)
                opening, _compact = self._selected_gripper_opening(obs, hand)
                last_gripper_opening = opening
                if opening <= threshold:
                    closed_streak += 1
                else:
                    closed_streak = 0
                closure_candidate = closed_streak >= required_closed_chunks
                task_success = official_task_success(info)

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
                    model_info=model_info,
                )
                states.append(state)
                self._current_observation = obs
                self._current_info = info

                # Environment stop flags are never evidence of a local grasp.
                if closure_candidate and not terminated and not truncated:
                    local_gripper_closure_detected = True
                    if self._local_grasp_validator is not None:
                        context = {
                            "hand": hand,
                            "instruction": instruction,
                            "chunk": chunks_used,
                            "env_steps": env_steps_used,
                            "selected_gripper_opening": opening,
                            "gripper_closed_threshold": threshold,
                            "consecutive_closed_chunks": closed_streak,
                            "required_closed_chunks": required_closed_chunks,
                            "task_success": task_success,
                        }
                        validator_result = bool(
                            self._local_grasp_validator(obs, context)
                        )
                        local_grasp_success = validator_result

                if task_success:
                    stop_reason = "task_success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                elif local_grasp_success:
                    stop_reason = "local_grasp_success"
                elif (
                    local_gripper_closure_detected
                    and self._local_grasp_validator is None
                ):
                    stop_reason = "local_gripper_closure_detected"
                elif env_steps_used >= self.max_episode_steps:
                    stop_reason = "horizon"
                elif chunks_used >= max_chunks:
                    stop_reason = "chunk_limit"
                else:
                    stop_reason = "running"
                _write_json_atomic(states_path, states)

            if stop_reason == "running":
                stop_reason = (
                    "horizon"
                    if env_steps_used >= self.max_episode_steps
                    else "chunk_limit"
                )
        except Exception as exc:
            logger.exception("pi0_pick failed")
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "error"

        result = {
            "_finish": True,
            "name": "pi0_pick",
            "hand": hand,
            "instruction": instruction,
            "model_instruction": model_instruction,
            "primitive_success": bool(local_grasp_success),
            "local_grasp_success": bool(local_grasp_success),
            "local_gripper_closure_detected": bool(
                local_gripper_closure_detected
            ),
            "visual_verification_required": True,
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
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "gripper_closed_threshold": gripper_closed_threshold,
            "required_closed_chunks": required_closed_chunks,
            "consecutive_closed_chunks": closed_streak,
            "final_selected_gripper_opening": last_gripper_opening,
            "elapsed_s": round(time.time() - started, 2),
            "states_path": str(states_path),
            "result_path": str(result_path),
            "action_trace_path": str(
                self.output_dir / "behavior_action_trace.jsonl"
            ),
            "manifest_path": str(self.output_dir / MANIFEST_FILENAME),
            "video_path": str(self.video_path),
            "video_fps": 15,
            "last_reward": _jsonable(last_reward),
            "last_info": _jsonable(last_info),
            "error": error,
        }
        _write_json_atomic(states_path, states)
        _write_json_atomic(result_path, result)
        self.last_result = result
        return result


__all__ = ["BehaviorPrimitives", "official_task_success"]
