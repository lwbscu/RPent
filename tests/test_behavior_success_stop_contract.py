from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robots.behavior.llm_preflight as llm_preflight
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.serial_vla_eval import run_vla_window_loop
from robots.behavior.tools import BehaviorPrimitives

_SUCCESS_ACTION = 5


def _observation() -> dict:
    return {
        "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros(256, dtype=np.float32),
        "task_descriptions": "pick up the trash and place it in the trash can",
    }


def _done(success: bool) -> dict:
    return {
        "done": {
            "success": success,
            "termination_conditions": {
                "timeout": {"done": False, "success": False},
                "predicate": {"done": success, "success": success},
            },
        }
    }


class _DirectProcess:
    def __init__(self) -> None:
        self.action_count = 0

    def step_env(self, _action, *, need_obs):
        assert need_obs is True
        self.action_count += 1
        if self.action_count > _SUCCESS_ACTION:
            raise AssertionError("an env action executed after raw task success")
        success = self.action_count == _SUCCESS_ACTION
        return (
            object(),
            np.array([float(success)]),
            np.array([success]),
            np.array([False]),
            [_done(success)],
        )


def _runtime(tmp_path) -> tuple[BehaviorEnvFacade, _DirectProcess]:
    direct = _DirectProcess()
    wrapped = {
        "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["pick up the trash and place it in the trash can"],
    }
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env = SimpleNamespace(
        _direct_process=direct,
        _wrap_obs=lambda _raw: wrapped,
    )
    facade._meta = {"max_episode_steps": 4096}
    facade._env_steps = 0
    facade._done = False
    facade._last_observation = _observation()
    facade._last_info = _done(False)
    facade._run_nonce = "run-success-stop"
    facade._attempt_nonce = "attempt-success-stop"
    facade._attempt_index = 1
    facade._official_success_latched = False
    facade._official_success_receipt = None
    facade._official_success_receipt_path = tmp_path / "official_success_receipt.json"
    facade._motion_frozen = False
    facade._motion_in_flight = False
    facade._controller_state = "vla"
    facade._action_source = "pi0_vla"
    facade._vla_actions_enabled = True
    facade._gripper_latch = {"left": 1.0, "right": 1.0}
    facade._planner_video_interval_steps = 4
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda *_args, **_kwargs: None
    facade._video_error = None
    facade._video_path = tmp_path / "episode.mp4"
    facade._video_sealed = False
    facade._finalize_video_segment = lambda: None
    facade._active_vla_invocation = "invocation-success-stop"
    facade._active_vla_call_index = 1
    facade._pending_vla_visual_authorization = None
    facade._pending_vla_attachment_snapshot = None
    facade._next_pi0_chunk_index = 1
    facade._latest_successful_held_rotate_receipt = None
    facade._latest_successful_held_rotate_attachment = None
    facade._latest_successful_held_rotate_public_frame_ids = set()
    facade._held_rotate_target_surface_review = None
    facade._awaiting_opposite_surface_review = None
    facade._completed_opposite_surface_cycles = []
    facade._active_rotate_pi0_candidate = None
    facade._sanitized_capability_summary = lambda: {
        "attachments": {
            "available": True,
            "count": 0,
            "by_hand": {
                "left": {"attached": False},
                "right": {"attached": False},
            },
            "conflict": False,
        },
        "gripper_state": {"left": "open", "right": "open"},
    }
    return facade, direct


class _Env:
    def __init__(self, facade: BehaviorEnvFacade) -> None:
        self.facade = facade
        self.chunk_step_calls = 0
        self.finalize_calls = 0

    @property
    def total_env_steps(self) -> int:
        return int(self.facade._env_steps)

    def prepare_vla_invocation(self, *, vla_status, **_kwargs):
        return {
            "primitive_success": True,
            "task_success": False,
            "failed_preconditions": [],
            "vla_actions_enabled": vla_status is not None,
            "attachment_count_at_invocation_start": 0,
            "attachments_present_at_invocation_start": False,
        }

    def current_observation(self):
        return self.facade._last_observation, {
            **_done(False),
            "_rpent": {"total_env_steps": self.total_env_steps},
        }

    def pi0_nav_pick_chunk_step(self, actions, *, chunk_index):
        self.chunk_step_calls += 1
        if self.chunk_step_calls > 1:
            raise AssertionError("a simulator chunk RPC was sent after task success")
        assert chunk_index == 1
        return self.facade._step_action_chunk(
            actions,
            observe_final=True,
            pi0_nav_pick=True,
        )

    def finalize_paused_runtime(self, vla_status):
        self.finalize_calls += 1
        assert vla_status["actions_enabled"] is False
        assert vla_status["healthz"]["actions_enabled"] is False
        receipt = self.facade._official_success_receipt
        assert isinstance(receipt, dict)
        self.facade._freeze_official_success_runtime()
        return {
            "controller_state": "frozen",
            "vla_actions_enabled": False,
            "lifecycle_finalized": True,
            "task_success": True,
            "official_success_receipt": receipt,
            "capability": self.facade._sanitized_capability_summary(),
        }


class _Model:
    endpoint = "http://127.0.0.1:9999"

    def __init__(self) -> None:
        self.actions_enabled = False
        self.predict_calls = 0

    def enable_actions(self):
        self.actions_enabled = True
        return {"actions_enabled": True, "pid": 123}

    def disable_actions(self):
        self.actions_enabled = False
        return {"actions_enabled": False, "pid": 123}

    def healthz(self):
        return {"actions_enabled": self.actions_enabled, "pid": 123}

    def predict_action_batch(self, observation, mode="eval"):
        assert observation["task_descriptions"]
        assert mode == "eval"
        self.predict_calls += 1
        if self.predict_calls > 1:
            raise AssertionError("a VLA predict was sent after raw task success")
        return np.zeros((32, 23), dtype=np.float32), {"model": "fake"}


def _primitives(tmp_path, *, pure: bool):
    facade, direct = _runtime(tmp_path)
    env = _Env(facade)
    model = _Model()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=4096,
        output_dir=tmp_path,
        initial_observation=_observation(),
        initial_info={
            **_done(False),
            "_rpent": {
                "run_nonce": "run-success-stop",
                "attempt_nonce": "attempt-success-stop",
                "attempt_index": 1,
                "total_env_steps": 0,
            },
        },
        task_name="picking_up_trash",
        public_seed=13,
        job_id="success-stop-contract",
        pure_vla_baseline=pure,
        max_tool_calls=None if pure else 350,
    )
    return primitives, env, model, direct


@pytest.mark.parametrize(
    ("controller", "requested_chunks"),
    (("agentic", 20), ("pure_vla", 80)),
)
def test_raw_success_mid_chunk_stops_all_further_actions_and_predictions(
    tmp_path,
    controller,
    requested_chunks,
):
    primitives, env, model, direct = _primitives(
        tmp_path / controller,
        pure=controller == "pure_vla",
    )

    if controller == "agentic":
        result = primitives.pi0_nav_pick(
            instruction="pick up the trash and place it in the trash can",
            chunks=requested_chunks,
        )
    else:
        loop = run_vla_window_loop(
            primitives,
            instruction="pick up the trash and place it in the trash can",
            deadline_monotonic=10_000.0,
            trace_path=tmp_path / "pure_tool_trace.jsonl",
            clock=lambda: 0.0,
        )
        assert loop.tool_calls == 1
        assert loop.stopped_reason == "official_task_success"
        result = loop.last_result

    assert isinstance(result, dict)
    assert direct.action_count == _SUCCESS_ACTION
    assert model.predict_calls == 1
    assert env.chunk_step_calls == 1
    assert env.finalize_calls == 1
    assert primitives._vla_invocations == 1
    assert result["task_success"] is True
    assert result["stop_reason"] == "official_task_success"
    assert result["requested_chunks"] == requested_chunks
    assert result["chunks_used"] == 1
    assert result["full_chunks_executed"] == 0
    assert result["vla_env_steps_used"] == _SUCCESS_ACTION
    assert result["env_steps_used"] == _SUCCESS_ACTION
    assert result["total_env_steps"] == _SUCCESS_ACTION
    assert result["exact_requested_chunks_completed"] is False
    assert result["error"] is None
    receipt = result["official_success_receipt"]
    assert receipt["env_step"] == _SUCCESS_ACTION
    assert receipt["raw_done"]["success"] is True

    states = json.loads(
        (
            tmp_path
            / controller
            / "vla_calls"
            / "call_001"
            / "pi0_nav_pick_states.json"
        ).read_text(encoding="utf-8")
    )
    assert [state["chunk"] for state in states] == [0, 1]
    assert states[-1]["pi0_nav_pick_monitor"]["executed_steps"] == _SUCCESS_ACTION
    state_receipt = states[-1]["official_success_receipt"]
    assert state_receipt["receipt_sha256"] == receipt["receipt_sha256"]
    assert state_receipt["env_step"] == receipt["env_step"]
    assert state_receipt["raw_done"] == receipt["raw_done"]


def test_llm_proxy_preflight_inherits_proxy_environment_without_task_context(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class _Popen:
        pid = 4242
        returncode = 0

        def __init__(self, command, **kwargs):
            captured["command"] = tuple(command)
            captured.update(kwargs)

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            challenge = captured["command"][
                captured["command"].index("--challenge") + 1
            ]
            payload = {
                "response": f"{llm_preflight.LLM_PREFLIGHT_TOKEN}:{challenge}",
                "completed_status": "completed",
                "errors": [],
                "forbidden_events": [],
            }
            return (
                "RPENT_LLM_PREFLIGHT_RESULT="
                + json.dumps(payload, ensure_ascii=True)
                + "\n",
                "",
            )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(llm_preflight.subprocess, "Popen", _Popen)
    monkeypatch.setattr(llm_preflight, "_worker_group_alive", lambda _pgid: False)
    environment = {
        "HTTP_PROXY": "http://127.0.0.1:7890",
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        "ALL_PROXY": "http://127.0.0.1:7890",
        "NO_PROXY": "127.0.0.1,localhost",
        "RPENT_BEHAVIOR_OUTPUT_DIR": "/must/not/leak",
    }

    receipt = llm_preflight.run_llm_proxy_preflight(
        python=sys.executable,
        repo_root=tmp_path,
        model="gpt-5.5",
        timeout_s=60,
        environment=environment,
    )

    captured_environment = captured["env"]
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        assert captured_environment[key] == environment[key]
    assert set(captured_environment) == {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "HOME",
        "CODEX_HOME",
    }
    isolated_home = Path(captured_environment["HOME"])
    isolated_codex_home = Path(captured_environment["CODEX_HOME"])
    assert isolated_home.name == "home"
    assert isolated_codex_home.name == "codex-home"
    assert isolated_home.parent == isolated_codex_home.parent
    assert "RPENT_BEHAVIOR_OUTPUT_DIR" not in captured_environment
    assert captured["cwd"] == tmp_path
    assert captured["start_new_session"] is True
    assert captured["timeout"] == 60
    assert captured["command"][:4] == (
        str(Path(sys.executable).absolute()),
        "-m",
        "robots.behavior.llm_preflight",
        "--worker",
    )
    assert receipt["status"] == "passed"
    assert receipt["model"] == "gpt-5.5"
    assert receipt["valid_response"] is True
    assert receipt["response_chars"] > len(llm_preflight.LLM_PREFLIGHT_TOKEN)
    assert receipt["response_sha256"]
    assert receipt["challenge_sha256"]
    assert receipt["transient_transport_events"] == 0
    assert receipt["failure_reason"] is None
    assert receipt["worker_returncode"] == 0
    assert receipt["outer_invocation_count"] == 1
    assert receipt["cleanup_verified"] is True
    assert receipt["isolation"] == {
        "ephemeral_thread": True,
        "tools_enabled": False,
        "task_context_supplied": False,
        "environment_rpc_supplied": False,
        "frozen_memory_supplied": False,
    }
