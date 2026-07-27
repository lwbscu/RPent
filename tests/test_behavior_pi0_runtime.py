import hashlib
import json

import numpy as np
import pytest

from robots.behavior.tools import BehaviorPrimitives


def _observation() -> dict:
    return {
        "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros(256, dtype=np.float32),
        "task_descriptions": "turn on the radio",
    }


def _receipt(*, env_step: int) -> dict:
    receipt = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": "run-one",
        "attempt_nonce": "attempt-one",
        "attempt_index": 1,
        "env_step": env_step,
        "raw_done": {"success": True},
    }
    unsigned = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return receipt


class _Model:
    endpoint = "http://127.0.0.1:9999"

    def __init__(self):
        self.enable_calls = 0
        self.disable_calls = 0
        self.predict_calls = 0
        self.actions_enabled = False

    def enable_actions(self):
        self.enable_calls += 1
        self.actions_enabled = True
        return {"actions_enabled": True, "pid": 123}

    def disable_actions(self):
        self.disable_calls += 1
        self.actions_enabled = False
        return {"actions_enabled": False, "pid": 123}

    def healthz(self):
        return {"actions_enabled": self.actions_enabled, "pid": 123}

    def predict_action_batch(self, observation, mode="eval"):
        self.predict_calls += 1
        assert observation["task_descriptions"]
        assert mode == "eval"
        return np.zeros((32, 23), dtype=np.float32), {"model": "fake"}


class _Env:
    def __init__(
        self,
        outcomes,
        *,
        preflight_failed_preconditions=None,
        current_observation_error=None,
        confirm_error=None,
        success_on_confirm=False,
    ):
        self.outcomes = list(outcomes)
        self.preflight_failed_preconditions = list(preflight_failed_preconditions or [])
        self.current_observation_error = current_observation_error
        self.confirm_error = confirm_error
        self.success_on_confirm = bool(success_on_confirm)
        self.total_env_steps = 0
        self.current_observation_calls = 0
        self.prepare_calls = []
        self.events = []
        self.finalize_calls = 0
        self.chunk_step_calls = 0
        self.invocation_chunk_step_calls = 0
        self.attachment_count = 0
        self.task_success_latched = False
        self.success_receipt = None

    def _info(self, *, task_success=False, monitor=None):
        raw_success = bool(task_success)
        info = {
            "done": {
                "success": raw_success,
                "termination_conditions": {
                    "predicate": {
                        "done": raw_success,
                        "success": raw_success,
                    },
                    "timeout": {"done": False, "success": False},
                },
            },
            "_rpent": {
                "run_nonce": "run-one",
                "attempt_nonce": "attempt-one",
                "attempt_index": 1,
                "total_env_steps": self.total_env_steps,
            },
        }
        if monitor is not None:
            info["_rpent"]["pi0_nav_pick_monitor"] = monitor
        return info

    def current_observation(self):
        self.current_observation_calls += 1
        self.events.append("current_observation")
        if self.current_observation_error is not None:
            raise RuntimeError(self.current_observation_error)
        return _observation(), self._info()

    def advance_planner_steps(self, count):
        self.total_env_steps += int(count)

    def prepare_vla_invocation(
        self,
        *,
        invocation_id,
        call_index,
        vla_status,
        current_object_visual_check=None,
    ):
        if vla_status is None:
            self.invocation_chunk_step_calls = 0
        self.prepare_calls.append((invocation_id, call_index, vla_status))
        self.events.append(
            "prepare_preflight" if vla_status is None else "prepare_confirm"
        )
        if vla_status is None and self.preflight_failed_preconditions:
            return {
                "primitive_success": False,
                "task_success": False,
                "failed_preconditions": list(self.preflight_failed_preconditions),
                "total_env_steps": self.total_env_steps,
            }
        if vla_status is not None and self.confirm_error is not None:
            raise RuntimeError(self.confirm_error)
        if vla_status is not None and self.success_on_confirm:
            self.task_success_latched = True
            self.success_receipt = _receipt(env_step=self.total_env_steps)
            return {
                "primitive_success": False,
                "task_success": True,
                "official_success_receipt": self.success_receipt,
                "failed_preconditions": ["official_success_latched"],
                "vla_actions_enabled": False,
                "attachment_count_at_invocation_start": self.attachment_count,
                "attachments_present_at_invocation_start": self.attachment_count > 0,
            }
        return {
            "primitive_success": True,
            "task_success": False,
            "failed_preconditions": [],
            "vla_actions_enabled": vla_status is not None,
            "attachment_count_at_invocation_start": self.attachment_count,
            "attachments_present_at_invocation_start": self.attachment_count > 0,
        }

    def pi0_nav_pick_chunk_step(self, actions, *, chunk_index):
        self.chunk_step_calls += 1
        self.invocation_chunk_step_calls += 1
        assert np.asarray(actions).shape == (32, 23)
        assert chunk_index == self.invocation_chunk_step_calls
        outcome = self.outcomes.pop(0)
        executed = int(outcome.get("executed_steps", 32))
        extra_env_steps = int(outcome.get("extra_env_steps", 0))
        self.total_env_steps += executed + extra_env_steps
        task_success = bool(outcome.get("task_success", False))
        local_grasp = bool(outcome.get("local_grasp_success", False))
        self.attachment_count = int(outcome.get("attachment_count", int(local_grasp)))
        left_attached = self.attachment_count >= 1
        right_attached = self.attachment_count >= 2
        receipt = _receipt(env_step=self.total_env_steps) if task_success else None
        if receipt is not None:
            self.task_success_latched = True
            self.success_receipt = receipt
        monitor = {
            "executed_steps": executed,
            "handoff_env_steps": int(
                outcome.get("reported_handoff_env_steps", extra_env_steps)
            ),
            "total_env_steps": self.total_env_steps,
            "local_grasp_success": local_grasp,
            "capability": {
                "attachments": {
                    "available": True,
                    "count": self.attachment_count,
                    "by_hand": {
                        "left": {"attached": left_attached},
                        "right": {"attached": right_attached},
                    },
                    "conflict": False,
                },
                "gripper_state": {
                    "left": "closed" if left_attached else "open",
                    "right": "closed" if right_attached else "open",
                },
            },
            "controller_state": "vla",
            "action_source": "pi0_vla",
            "vla_actions_enabled": True,
            "official_success_receipt": self.success_receipt,
            "stop_reason": (
                None
                if outcome.get("continue_vla", False)
                else outcome.get("stop_reason", "runtime_test_boundary")
            ),
            # Deliberately incomplete and non-existent: audit images cannot gate.
            "visual_review": {"metadata_path": "/does/not/exist.json"},
        }
        info = self._info(task_success=task_success, monitor=monitor)
        if receipt is not None:
            info["_rpent"]["official_success_receipt"] = receipt
        return (
            _observation(),
            1.0 if task_success else 0.0,
            bool(outcome.get("terminated", False)),
            bool(outcome.get("truncated", False)),
            info,
        )

    def finalize_paused_runtime(self, vla_status):
        self.finalize_calls += 1
        assert vla_status["actions_enabled"] is False
        assert vla_status["healthz"]["actions_enabled"] is False
        return {
            "controller_state": ("frozen" if self.task_success_latched else "planner"),
            "vla_actions_enabled": False,
            "lifecycle_finalized": True,
            "task_success": self.task_success_latched,
            "official_success_receipt": self.success_receipt,
            "capability": {
                "attachments": {
                    "available": True,
                    "count": self.attachment_count,
                    "by_hand": {
                        "left": {"attached": self.attachment_count >= 1},
                        "right": {"attached": self.attachment_count >= 2},
                    },
                    "conflict": False,
                },
                "gripper_state": {
                    "left": "closed" if self.attachment_count >= 1 else "open",
                    "right": "closed" if self.attachment_count >= 2 else "open",
                },
            },
        }


def _primitives(
    tmp_path,
    outcomes,
    *,
    preflight_failed_preconditions=None,
    current_observation_error=None,
    confirm_error=None,
    success_on_confirm=False,
    pure_vla_baseline=False,
    max_tool_calls=350,
):
    initial_info = {
        "done": {"success": False},
        "_rpent": {
            "run_nonce": "run-one",
            "attempt_nonce": "attempt-one",
            "attempt_index": 1,
            "total_env_steps": 0,
        },
    }
    model = _Model()
    env = _Env(
        outcomes,
        preflight_failed_preconditions=preflight_failed_preconditions,
        current_observation_error=current_observation_error,
        confirm_error=confirm_error,
        success_on_confirm=success_on_confirm,
    )
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=256,
        output_dir=tmp_path,
        initial_observation=_observation(),
        initial_info=initial_info,
        job_id="job-one",
        pure_vla_baseline=pure_vla_baseline,
        max_tool_calls=max_tool_calls,
    )
    return primitives, env, model


def test_raw_success_between_prepare_phases_disables_vla_without_inference(
    tmp_path,
):
    primitives, env, model = _primitives(
        tmp_path,
        [],
        success_on_confirm=True,
    )

    result = primitives.pi0_nav_pick(
        instruction="continue manipulating the radio",
        chunks=1,
    )

    assert result["task_success"] is True
    assert result["stop_reason"] == "official_task_success"
    assert result["chunks_used"] == 0
    assert result["env_steps_used"] == 0
    assert result["official_success_receipt"]["source"] == 'info["done"]["success"]'
    assert model.enable_calls == 1
    assert model.disable_calls == 1
    assert model.predict_calls == 0
    assert env.current_observation_calls == 0
    assert env.chunk_step_calls == 0
    assert env.finalize_calls == 1


def test_no_grasp_returns_safely_and_next_call_rearms_vla(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [
            {"local_grasp_success": False},
            {"local_grasp_success": False},
        ],
    )

    first = primitives.pi0_nav_pick(instruction="search for the radio", chunks=1)
    second = primitives.pi0_nav_pick(
        instruction="search from another view",
        chunks=1,
    )

    assert first["primitive_success"] is True
    assert second["primitive_success"] is True
    assert first["_finish"] is False
    assert second["_finish"] is False
    assert first["stop_reason"] == second["stop_reason"] == "requested_chunks_completed"
    assert model.enable_calls == 2
    assert model.disable_calls == 2
    assert env.finalize_calls == 2
    assert len(env.prepare_calls) == 4
    assert env.current_observation_calls == 2
    assert (tmp_path / "vla_calls" / "call_001").is_dir()
    assert (tmp_path / "vla_calls" / "call_002").is_dir()


def test_pure_vla_primitives_pass_private_authorization_only_in_baseline(
    tmp_path,
) -> None:
    primitives, env, _model = _primitives(
        tmp_path,
        [{"local_grasp_success": False}],
        pure_vla_baseline=True,
        max_tool_calls=None,
    )
    original = env.prepare_vla_invocation
    captured: list[dict[str, object]] = []

    def prepare(**kwargs):
        captured.append(dict(kwargs))
        assert kwargs.pop("baseline_internal_authorization") is True
        return original(**kwargs)

    env.prepare_vla_invocation = prepare

    result = primitives.pi0_nav_pick(
        instruction="complete the authoritative task",
        chunks=1,
    )

    assert result["exact_requested_chunks_completed"] is True
    assert len(captured) == 2
    assert all("current_object_visual_check" not in arguments for arguments in captured)


def test_unlimited_tool_calls_are_reserved_for_pure_vla_baseline(tmp_path) -> None:
    with pytest.raises(ValueError, match="pure VLA baseline"):
        BehaviorPrimitives(
            env=None,
            model=None,
            output_dir=tmp_path / "hybrid",
            max_tool_calls=None,
            pure_vla_baseline=False,
        )

    baseline = BehaviorPrimitives(
        env=None,
        model=None,
        output_dir=tmp_path / "baseline",
        max_tool_calls=None,
        pure_vla_baseline=True,
    )
    assert baseline.max_tool_calls is None


def test_stable_dual_attachments_run_beyond_128_chunks_until_episode_boundary(
    tmp_path,
):
    outcomes = [
        {
            "local_grasp_success": True,
            "attachment_count": 2,
            "continue_vla": True,
        }
        for _ in range(129)
    ]
    primitives, env, model = _primitives(tmp_path, outcomes)
    env.attachment_count = 2
    primitives.max_episode_steps = 129 * 32

    result = primitives.pi0_nav_pick(
        instruction="continue the task with both held objects",
        chunks=129,
        current_object_visual_check={
            "camera": "head",
            "frame_id": "head:0:fresh",
            "assessment": "current_task_object_configuration_reviewed",
        },
    )

    assert result["error"] is None
    assert result["primitive_success"] is True
    assert result["stop_reason"] == "requested_chunks_completed"
    assert result["requested_chunks"] == 129
    assert result["exact_requested_chunks_completed"] is True
    assert result["chunks_used"] == 129
    assert result["full_chunks_executed"] == 129
    assert result["vla_env_steps_used"] == 129 * 32
    assert result["global_vla_chunks"] == 129
    assert "max_chunks" not in result
    assert "max_total_vla_chunks" not in result
    assert model.predict_calls == 129
    assert env.chunk_step_calls == 129
    assert env.finalize_calls == 1

    call_record = json.loads(
        (tmp_path / "vla_calls" / "call_001" / "pi0_nav_pick_call.json").read_text(
            encoding="utf-8"
        )
    )
    assert call_record["schema_version"] == 5
    assert call_record["requested_chunks"] == 129
    assert "max_chunks" not in call_record
    assert "max_total_vla_chunks" not in call_record
    assert "max_vla_chunks_per_call" not in call_record


def test_pi0_vla_actions_are_forwarded_without_literal_hand_rewrite(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [{"local_grasp_success": False}],
    )
    expected = np.arange(32 * 23, dtype=np.float32).reshape(32, 23) * np.float32(0.001)
    instruction = "preserve this exact instruction; left/right are VLA-owned"
    model_inputs = []
    chunk_inputs = []

    def predict_action_batch(observation, mode="eval"):
        model_inputs.append((observation["task_descriptions"], mode))
        return expected.copy(), {"model": "sentinel"}

    original_chunk_step = env.pi0_nav_pick_chunk_step

    def record_chunk_step(actions, *, chunk_index):
        chunk_inputs.append(np.asarray(actions).copy())
        return original_chunk_step(actions, chunk_index=chunk_index)

    model.predict_action_batch = predict_action_batch
    env.pi0_nav_pick_chunk_step = record_chunk_step

    result = primitives.pi0_nav_pick(instruction=instruction, chunks=1)

    assert result["error"] is None
    assert model_inputs == [(instruction, "eval")]
    assert len(chunk_inputs) == 1
    assert chunk_inputs[0].shape == (32, 23)
    assert chunk_inputs[0].dtype == expected.dtype
    assert chunk_inputs[0].tobytes() == expected.tobytes()


def test_held_visual_authorization_precedes_private_observation_refresh(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [{"local_grasp_success": False}],
    )
    visual_check = {
        "camera": "head",
        "frame_id": "head:0:fresh",
        "assessment": "current_task_object_configuration_reviewed",
    }

    result = primitives.pi0_nav_pick(
        instruction="continue manipulating the held radio",
        chunks=1,
        current_object_visual_check=visual_check,
    )

    assert result["error"] is None
    assert env.events[:3] == [
        "prepare_preflight",
        "prepare_confirm",
        "current_observation",
    ]
    assert model.enable_calls == 1
    assert env.prepare_calls[0][2] is None
    assert env.prepare_calls[1][2]["actions_enabled"] is True


def test_vla_preflight_failure_is_public_and_does_not_refresh_observation(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [],
        preflight_failed_preconditions=["fresh_object_visual_check_required"],
    )

    result = primitives.pi0_nav_pick(
        instruction="continue manipulating the held radio",
        chunks=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "vla_runtime_precondition_rejected"
    assert result["failed_preconditions"] == ["fresh_object_visual_check_required"]
    assert result["chunks_used"] == 0
    assert env.events == ["prepare_preflight"]
    assert env.current_observation_calls == 0
    assert model.enable_calls == 0


def test_opposite_surface_latch_rejects_pi0_before_model_or_env_step(tmp_path):
    failure = "pi0_nav_pick_disabled_by_opposite_surface_receipt"
    primitives, env, model = _primitives(
        tmp_path,
        [],
        preflight_failed_preconditions=[failure],
    )

    result = primitives.pi0_nav_pick(
        instruction="continue manipulating the held radio",
        chunks=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "vla_runtime_precondition_rejected"
    assert result["failed_preconditions"] == [failure]
    assert result["chunks_used"] == 0
    assert result["env_steps_used"] == 0
    assert result["total_env_steps"] == 0
    assert env.events == ["prepare_preflight"]
    assert env.current_observation_calls == 0
    assert env.chunk_step_calls == 0
    assert model.enable_calls == 0
    assert model.predict_calls == 0


def test_private_observation_failure_disables_vla_and_finalizes_env(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [],
        current_observation_error="sensor refresh failed",
    )

    result = primitives.pi0_nav_pick(
        instruction="continue toward the radio",
        chunks=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "sensor refresh failed" in result["error"]
    assert env.events == [
        "prepare_preflight",
        "prepare_confirm",
        "current_observation",
    ]
    assert model.enable_calls == 1
    assert model.disable_calls == 1
    assert model.actions_enabled is False
    assert env.finalize_calls == 1


def test_rearm_confirmation_failure_also_disables_vla_and_finalizes_env(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [],
        confirm_error="confirmation failed",
    )

    result = primitives.pi0_nav_pick(
        instruction="continue toward the radio",
        chunks=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "confirmation failed" in result["error"]
    assert env.events == ["prepare_preflight", "prepare_confirm"]
    assert env.current_observation_calls == 0
    assert model.enable_calls == 1
    assert model.disable_calls == 1
    assert model.actions_enabled is False
    assert env.finalize_calls == 1


def test_planner_steps_between_pi0_calls_rebase_handoff_accounting(tmp_path):
    primitives, env, _model = _primitives(
        tmp_path,
        [
            {"local_grasp_success": False},
            {"local_grasp_success": False},
        ],
    )

    first = primitives.pi0_nav_pick(instruction="search for the radio", chunks=1)
    env.advance_planner_steps(7)
    second = primitives.pi0_nav_pick(
        instruction="search from another view",
        chunks=1,
    )

    assert first["error"] is None
    assert first["total_env_steps"] == 32
    assert second["error"] is None
    assert second["env_steps_used"] == 32
    assert second["vla_env_steps_used"] == 32
    assert second["handoff_env_steps_used"] == 0
    assert second["total_env_steps"] == 71
    states = json.loads(
        (tmp_path / "vla_calls" / "call_002" / "pi0_nav_pick_states.json").read_text(
            encoding="utf-8"
        )
    )
    assert states[0]["chunk"] == 0
    assert states[0]["env_steps"] == 0
    assert states[0]["total_env_steps"] == 39
    assert states[1]["env_steps"] == 32
    assert states[1]["total_env_steps"] == 71


def test_fresh_baseline_enforces_full_chunk_budget_before_rearming_vla(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [
            {"local_grasp_success": False},
            {"local_grasp_success": False},
        ],
    )
    primitives.max_episode_steps = 64

    first = primitives.pi0_nav_pick(instruction="search for the radio", chunks=1)
    env.advance_planner_steps(1)
    second = primitives.pi0_nav_pick(
        instruction="search from another view",
        chunks=1,
    )

    assert first["total_env_steps"] == 32
    assert second["error"] is None
    assert second["stop_reason"] == "requested_chunks_exceed_remaining_episode_steps"
    assert second["chunks_used"] == 0
    assert second["total_env_steps"] == 33
    assert model.enable_calls == 1
    assert env.current_observation_calls == 1
    states = json.loads(
        (tmp_path / "vla_calls" / "call_002" / "pi0_nav_pick_states.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(states) == 1
    assert states[0]["env_steps"] == 0
    assert states[0]["total_env_steps"] == 33


def test_accounting_error_preserves_completed_chunk_audit(tmp_path):
    primitives, _env, _model = _primitives(
        tmp_path,
        [
            {
                "local_grasp_success": False,
                "extra_env_steps": 2,
                "reported_handoff_env_steps": 0,
            }
        ],
    )

    result = primitives.pi0_nav_pick(
        instruction="search for the radio",
        chunks=1,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "error"
    assert "handoff accounting mismatch" in result["error"]
    assert result["chunks_used"] == 1
    assert result["full_chunks_executed"] == 1
    assert result["vla_env_steps_used"] == 32
    assert result["env_steps_used"] == 34
    assert result["total_env_steps"] == 34
    states = json.loads(
        (tmp_path / "vla_calls" / "call_001" / "pi0_nav_pick_states.json").read_text(
            encoding="utf-8"
        )
    )
    assert [state["chunk"] for state in states] == [0, 1]
    assert states[1]["env_steps"] == 34
    assert states[1]["total_env_steps"] == 34


def test_local_grasp_uses_runtime_capability_and_transfers_controller(
    tmp_path,
):
    primitives, env, model = _primitives(
        tmp_path,
        [
            {
                "local_grasp_success": True,
                "attachment_count": 2,
                "executed_steps": 32,
            }
        ],
    )

    result = primitives.pi0_nav_pick(instruction="grasp the radio", chunks=1)

    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["_finish"] is False
    assert result["stop_reason"] == "requested_chunks_completed"
    assert "held_object_state" not in result
    assert result["env_steps_used"] == 32
    assert result["vla_env_steps_used"] % 32 == 0
    assert result["full_chunks_executed"] == result["chunks_used"] == 1
    assert model.disable_calls == 1
    assert env.finalize_calls == 1


def test_partial_chunk_cannot_claim_attachment_handoff(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [{"local_grasp_success": True, "executed_steps": 7}],
    )

    result = primitives.pi0_nav_pick(instruction="grasp the radio", chunks=1)

    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["stop_reason"] == "error"
    assert "without a terminal reason" in result["error"]
    assert model.disable_calls == 1
    assert model.actions_enabled is False
    assert env.finalize_calls == 1


def test_raw_official_success_returns_without_visual_or_controller_gate(tmp_path):
    primitives, env, model = _primitives(
        tmp_path,
        [{"task_success": True, "local_grasp_success": False}],
    )

    result = primitives.pi0_nav_pick(instruction="turn on the radio", chunks=1)

    assert result["task_success"] is True
    assert result["primitive_success"] is True
    assert result["_finish"] is True
    assert result["runner_termination_reason"] == "official_task_success"
    assert result["stop_reason"] == "official_task_success"
    assert result["official_success_receipt"]["raw_done"]["success"] is True
    assert model.disable_calls == 1
    assert env.finalize_calls == 1


class _NoRpc:
    def __getattr__(self, name):
        raise AssertionError(f"invalid input reached RPC: {name}")


class _MoveRpc:
    def __init__(self):
        self.calls = []

    def move_to(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "reached",
            "official_success_source": 'info["done"]["success"]',
        }


@pytest.mark.parametrize("depth_window_px", [0, 32])
def test_pixel_to_world_rejects_out_of_schema_window_before_rpc(
    tmp_path, depth_window_px
):
    primitives = BehaviorPrimitives(env=_NoRpc(), output_dir=tmp_path)
    with pytest.raises(ValueError, match=r"\[1,31\]"):
        primitives.pixel_to_world(
            camera="head",
            frame_id="frame",
            u=1,
            v=1,
            depth_window_px=depth_window_px,
        )


def test_move_to_has_no_fixed_upper_safety_thresholds(tmp_path):
    env = _MoveRpc()
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)

    result = primitives.move_to(
        hand="left",
        target={"projection_id": "p", "standoff_m": 0.31},
        visual_hand_check={
            "camera": "head",
            "frame_id": "head:0:fresh",
            "selected_hand": "left",
            "assessment": "selected_hand_visually_confirmed",
        },
        position_tolerance_m=0.051,
        max_travel_m=0.31,
    )

    assert result["primitive_success"] is True
    assert env.calls == [
        {
            "hand": "left",
            "target": {"projection_id": "p", "standoff_m": 0.31},
            "visual_hand_check": {
                "camera": "head",
                "frame_id": "head:0:fresh",
                "selected_hand": "left",
                "assessment": "selected_hand_visually_confirmed",
            },
            "position_tolerance_m": 0.051,
            "max_travel_m": 0.31,
                "timeout_s": 240.0,
        }
    ]
