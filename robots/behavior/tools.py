"""The single BEHAVIOR-specific full-task tool."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

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


class FullTaskRunner:
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
        self.output_dir = Path(output_dir) if output_dir else get_output_dir()
        self.video_path = (
            Path(video_path) if video_path else self.output_dir / "episode.mp4"
        )
        self.last_result: dict[str, Any] | None = None

    def _prepare_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("states.json", "final_result.json"):
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

            chunk_limit = self.max_chunks
            if chunk_limit is None:
                chunk_limit = math.ceil(self.max_episode_steps / self.action_horizon)

            while (
                not task_success
                and not terminated
                and not truncated
                and chunks_used < chunk_limit
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
                elif chunks_used >= chunk_limit:
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


__all__ = ["FullTaskRunner", "official_task_success"]
