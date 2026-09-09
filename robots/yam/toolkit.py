# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent tools for the real YAM robot."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.yam import tools
from robots.yam.contracts import YAM_CAMERA_NAMES
from robots.yam.primitives import YamPrimitives
from robots.yam.projection import world_from_depth as _world_from_depth_cv
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


_RECIPE_ACTIONS = {
    "pi05_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
}


class YamToolkit(Toolkit):
    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}
    _FRAME_ARTIFACTS = {
        "top": "top_rgb.png",
        "left": "left_rgb.png",
        "right": "right_rgb.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        run_output_dir: Path | str | None = None,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported YAM toolkit mode: {mode!r}")
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._mode = mode
        self._attempt = 1
        self._attempts_per_session = max(0, int(attempts_per_session))
        self._session_attempt = 1
        self._run_output_dir = Path(
            run_output_dir or state_output_dir or get_output_dir()
        )
        self._latest_status: dict[str, Any] = {}
        self._primitives = YamPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._register_yam_tools()
        initial = self.get_env_state(
            command={"action": "observe"},
            result={"success": True},
            elapsed_s=0.0,
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        initial_state = initial.get("state")
        if isinstance(initial_state, dict):
            self._latest_status = initial_state.get(
                "episode_status", self._latest_status
            )

    def _register_yam_tools(self) -> None:
        self._tools.pop("finish", None)
        self.add_tool(
            "view_env_state",
            self._SPECS["view_env_state"],
            partial(tools.view_env_state, state=self._state),
        )
        self.add_tool(
            "sample_world_xyz",
            self._SPECS["sample_world_xyz"],
            partial(tools.sample_world_xyz, self._state),
        )
        self.add_tool(
            "query_world_map",
            self._SPECS["query_world_map"],
            partial(tools.query_world_map, self._state),
        )
        for name in (
            "render",
            "move_to",
            "rotate_wrist",
            "set_gripper",
            "release",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        if self._primitives.model is not None:
            self.add_tool(
                "pi05_act", self._SPECS["pi05_act"], partial(self._step, "pi05_act")
            )
        if self._mode == "exploration":
            self.add_tool("reset", self._SPECS["reset"], self._reset_episode)
        self.add_tool("finish", self._SPECS["finish"], self._finish)

    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return {"_finish": True, "status": status, "summary": summary}

    def _reset_episode(self) -> dict[str, Any]:
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": f"This session's attempt budget is spent ({budget} attempts).",
            }
        result = self._primitives.reset()
        self._attempt += 1
        self._session_attempt += 1
        self._latest_status = result.get("episode_status", {})
        result["attempt"] = self._attempt
        result["notice"] = (
            "YAM episode reset completed; re-run perception before acting."
        )
        return result

    def _capture_full_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._primitives.env
        obs, info = env.observe()
        views: dict[str, dict[str, Any]] = {}
        for camera_name in YAM_CAMERA_NAMES:
            source = obs.get("views", {}).get(camera_name)
            if not isinstance(source, dict):
                raise TypeError(f"YAM observe.views is missing {camera_name!r}")
            rgb = source.get("rgb")
            depth = source.get("depth")
            camera_meta = source.get("camera_meta")
            if rgb is None or depth is None or not isinstance(camera_meta, dict):
                raise TypeError(
                    f"YAM observe.views[{camera_name!r}] must contain rgb, depth, camera_meta"
                )
            projection_error = None
            try:
                world = _world_from_depth_cv(depth, camera_meta)
            except (ValueError, RuntimeError) as error:
                world = np.full((*np.asarray(depth).shape, 3), np.nan, dtype=np.float32)
                projection_error = str(error)
            views[camera_name] = {
                "rgb": np.asarray(rgb),
                "depth": np.asarray(depth, dtype=np.float32),
                "world_xyz": world,
                "camera_meta": camera_meta,
            }
            if projection_error:
                views[camera_name]["world_xyz_limitation"] = projection_error
        return {
            "views": views,
            "robot_state": info["robot_state"],
            "task_name": env.server_meta["task_name"],
            "task_language": env.get_task_language(),
            "depth_unit": "metres",
            "world_frame": "left_base",
            "snapshot_id": obs.get("snapshot_id"),
        }, info["episode_status"]

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        observation, status = self._capture_full_observation()
        self._latest_status = status
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        captured = tools.view_env_state(record.step_idx, state=self._state)
        if result.get("_finish"):
            # Toolkit.execute_tool replaces stateful handler output with this
            # capture. Keep the planner termination signal at the top level.
            verified = status.get("eval_success") is True
            requested_status = str(result.get("status", "failure"))
            captured.update(
                {
                    **result,
                    "requested_status": requested_status,
                    "requested_success": requested_status.lower() == "success",
                    "verified_success": verified,
                    "episode_status": status,
                    "status": "success" if verified else "failure",
                }
            )
        return captured

    def cancel_active_and_wait(self) -> None:
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        self._primitives.env.request_stop()
        operation.done_event.wait()

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def close(self) -> None:
        stop_error = None
        try:
            self._primitives.env.request_stop()
        except Exception as error:
            stop_error = error
        frames = self._primitives.stop_recording()
        if frames:
            self._state.save("episode.mp4", frames, step=None, fps=20)
        if stop_error:
            raise RuntimeError(
                "Could not deliver YAM stop request during toolkit close"
            ) from stop_error

    def solved(self) -> bool:
        return self._latest_status.get("eval_success") is True

    def write_recipe(self, recipe_tag: str) -> str:
        if not self.solved():
            return ""
        episode_id = self._latest_status["episode_id"]
        successful_records = [
            record
            for record in self._state.records()
            if record.state["episode_status"]["episode_id"] == episode_id
        ]
        if not any(record.terminated for record in successful_records):
            return ""
        recipe = []
        for record in successful_records:
            command = record.command
            result = record.result
            if (
                not isinstance(command, dict)
                or command.get("action") not in _RECIPE_ACTIONS
            ):
                continue
            if isinstance(result, dict) and (
                result.get("error") or result.get("success") is False
            ):
                continue
            recipe.append(command)
        self._run_output_dir.mkdir(parents=True, exist_ok=True)
        name = f"{recipe_tag}_recipe.jsonl"
        recipe_path = self._run_output_dir / name
        recipe_path.write_text(
            "".join(
                json.dumps(command, ensure_ascii=False) + "\n" for command in recipe
            ),
            encoding="utf-8",
        )
        audit_path = self._run_output_dir / f"{recipe_tag}.json"
        audit = {
            "robot": "yam",
            "recipe_tag": recipe_tag,
            "task_name": successful_records[-1].state.get("task_name")
            if successful_records
            else None,
            "task_language": successful_records[-1].state.get("task_language")
            if successful_records
            else None,
            "solved": True,
            "source": "operator_backed_eval_success",
            "episode_status": dict(self._latest_status),
            "terminal_step": successful_records[-1].step_idx
            if successful_records
            else None,
            "recipe_actions": len(recipe),
            "session_state_dir": str(self._state_output_dir),
            "coordinate_scope": "episode-local; re-localize before reuse",
        }
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(recipe_path)
