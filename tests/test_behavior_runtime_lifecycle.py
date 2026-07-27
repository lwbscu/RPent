from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import robots.behavior.env_server as env_server_module
from robots.behavior.camera_geometry import (
    CameraCorrectionProfile,
    CameraGeometryError,
    CameraIntrinsics,
    FrameCache,
    r1pro_wrist_camera_reference_transforms,
)
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import (
    _CONTROLLER_VLA,
    _ENV_RPC_METHODS,
    BehaviorEnvFacade,
    _flush_shutdown_artifacts,
    _MainThreadDispatcher,
)
from robots.behavior.schemas import RAW_PROPRIO_SEGMENTS, ROTATE_WRIST_RUNTIME_TIMEOUT_S
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
)
from robots.behavior.terminal_success import summarize_action_trace_success


def _runtime() -> BehaviorEnvFacade:
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._reset_completed = True
    facade._controller_mode = "hybrid"
    facade._motion_frozen = False
    facade._motion_in_flight = False
    facade._official_success_latched = False
    facade._last_info = {"done": {"success": False}}
    facade._controller_state = _CONTROLLER_VLA
    facade._base_controller_mode = "velocity"
    facade._action_source = "pi0_vla"
    facade._vla_actions_enabled = True
    facade._active_vla_invocation = None
    facade._active_vla_call_index = None
    facade._pending_vla_visual_authorization = None
    facade._pending_vla_attachment_snapshot = None
    facade._pending_vla_baseline_internal_authorization = False
    facade._latest_successful_held_rotate_receipt = None
    facade._latest_successful_held_rotate_attachment = None
    facade._latest_successful_held_rotate_public_frame_ids = set()
    facade._held_rotate_target_surface_review = None
    facade._active_rotate_pi0_candidate = None
    facade._awaiting_opposite_surface_review = None
    facade._completed_opposite_surface_cycles = []
    facade._pi0_nav_pick_disable_receipt = None
    facade._public_capture_sequence = 0
    facade._latest_unconsumed_public_capture_receipt = None
    facade._next_pi0_chunk_index = 1
    facade._attempt_index = 1
    facade._attempt_nonce = "attempt"
    facade._run_nonce = "run"
    facade._env_steps = 0
    facade._projection_receipts = {}
    facade._consumed_projection_receipts = set()
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    facade._public_observed_frame_ids = {"head:0:visual-selection"}
    facade._latest_public_head_frame_id = "head:0:visual-selection"
    facade._held_closure_receipts = {}
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: SimpleNamespace(
            frame_id=frame_id,
            step_index=0,
            capture_group_id="capture:0:visual-selection",
        )
    )
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 0,
        "hands": [],
        "identity_conflict": False,
        "attached_objects": {},
        "by_hand": {
            "left": {"attached": False},
            "right": {"attached": False},
        },
    }
    facade._switch_controller = lambda target, **_kwargs: {
        "from": facade._controller_state,
        "to": target,
        "changed": target != facade._controller_state,
    }
    return facade


def _attachment_facts(
    *,
    left=None,
    right=None,
    available: bool = True,
    identity_conflict: bool = False,
) -> dict:
    attached_objects = {
        hand: roots
        for hand, roots in (("left", left), ("right", right))
        if roots is not None
    }
    hands = [hand for hand in ("left", "right") if hand in attached_objects]
    return {
        "available": available,
        "attachment_count": len(hands),
        "hands": hands,
        "identity_conflict": identity_conflict,
        "attached_objects": attached_objects,
        "by_hand": {
            hand: {"attached": hand in attached_objects} for hand in ("left", "right")
        },
    }


def test_internal_initial_reset_remains_rpc_but_is_not_a_task_primitive():
    assert "reset" in _ENV_RPC_METHODS
    task_primitives = {
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
    }
    assert "reset" not in task_primitives


def test_pi0_can_be_rearmed_for_multiple_safe_invocations():
    facade = _runtime()
    first = facade.prepare_vla_invocation(
        invocation_id="call-1", call_index=1, vla_status=None
    )
    assert first["primitive_success"] is True
    facade._clear_active_vla_invocation_state()
    second = facade.prepare_vla_invocation(
        invocation_id="call-2", call_index=2, vla_status=None
    )
    assert second["primitive_success"] is True
    assert facade._active_vla_invocation == "call-2"


def test_pi0_rearm_requires_one_fresh_visual_check_when_attached():
    facade = _runtime()
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 1,
        "hands": ["right"],
        "identity_conflict": False,
        "attached_objects": {},
    }
    missing = facade.guard_tool_call(name="pi0_nav_pick", input_dict={})
    assert "fresh_object_visual_check_required" in missing["failed_preconditions"]
    visual_check = {
        "camera": "head",
        "frame_id": "head:0:fresh",
        "assessment": "current_task_object_configuration_reviewed",
    }

    def authorize(check, *, invocation_id):
        if check != visual_check:
            raise RuntimeError("fresh object visual review required")
        return {
            "invocation_id": invocation_id,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "env_step": 0,
        }

    facade._current_object_visual_authorization = authorize
    allowed = facade.guard_tool_call(
        name="pi0_nav_pick",
        input_dict={"current_object_visual_check": visual_check},
    )
    assert allowed["failed_preconditions"] == []
    preflight = facade.prepare_vla_invocation(
        invocation_id="held-call",
        call_index=1,
        vla_status=None,
        current_object_visual_check=visual_check,
    )
    assert preflight["primitive_success"] is True
    assert preflight["attachments_present_at_invocation_start"] is True
    assert preflight["attachment_count_at_invocation_start"] == 1


def test_trash_pi0_rearm_accepts_adaptive_chunks_with_fresh_review():
    facade = _runtime()
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/left-object")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/right-object")},
    )
    visual_check = {
        "camera": "head",
        "frame_id": "head:0:fresh",
        "assessment": "current_task_object_configuration_reviewed",
    }

    def authorize_dual(check, *, invocation_id):
        if check != visual_check:
            raise RuntimeError("fresh object visual review required")
        return {
            "invocation_id": invocation_id,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "env_step": 0,
        }

    facade._current_object_visual_authorization = authorize_dual
    missing = facade.guard_tool_call(
        name="pi0_nav_pick",
        input_dict={"chunks": 24},
    )
    assert missing["primitive_success"] is False
    assert missing["failed_preconditions"] == ["fresh_object_visual_check_required"]

    for chunks in (1, 24, 80):
        allowed = facade.guard_tool_call(
            name="pi0_nav_pick",
            input_dict={
                "chunks": chunks,
                "current_object_visual_check": visual_check,
            },
        )
        assert allowed["primitive_success"] is True
        assert allowed["failed_preconditions"] == []

    preflight = facade.prepare_vla_invocation(
        invocation_id="dual-call",
        call_index=1,
        vla_status=None,
        current_object_visual_check=visual_check,
    )
    assert preflight["primitive_success"] is True
    assert preflight["attachment_count_at_invocation_start"] == 2


def test_hybrid_runtime_rejects_private_baseline_authorization() -> None:
    facade = _runtime()
    facade._controller_mode = "hybrid"

    with pytest.raises(RuntimeError, match="unavailable in hybrid mode"):
        facade.prepare_vla_invocation(
            invocation_id="private-baseline",
            call_index=1,
            vla_status=None,
            baseline_internal_authorization=True,
        )


def test_pure_vla_runtime_uses_internal_capture_for_dual_attachments() -> None:
    facade = _runtime()
    facade._controller_mode = "pi0_nav_pick_only"
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/can")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/ashcan")},
    )
    captured: list[tuple[str, dict[str, object]]] = []

    def authorize(*, invocation_id, attachment_snapshot):
        captured.append((invocation_id, attachment_snapshot))
        return {
            "schema_version": 1,
            "source": "baseline_internal_synchronized_capture",
            "controller_mode": "pi0_nav_pick_only",
            "invocation_id": invocation_id,
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "env_step": 0,
            "resolved_camera": "head",
            "frame_id": "head:0:internal",
            "capture_group_id": "capture:0:internal",
            "assessment": "runtime_synchronized_state_only",
        }

    facade._baseline_internal_visual_authorization = authorize

    result = facade.prepare_vla_invocation(
        invocation_id="baseline-dual",
        call_index=1,
        vla_status=None,
        baseline_internal_authorization=True,
    )

    assert result["primitive_success"] is True
    assert result["baseline_internal_authorization"] is True
    assert result["attachment_count_at_invocation_start"] == 2
    authorization = result["current_object_visual_authorization"]
    assert authorization["source"] == "baseline_internal_synchronized_capture"
    assert authorization["attachment_count"] == 2
    assert captured[0][0] == "baseline-dual"
    assert captured[0][1]["hands"] == ["left", "right"]

    confirmed = facade.prepare_vla_invocation(
        invocation_id="baseline-dual",
        call_index=1,
        vla_status={"actions_enabled": True},
        baseline_internal_authorization=True,
    )
    assert confirmed["primitive_success"] is True
    assert confirmed["attachment_count_at_invocation_start"] == 2
    assert confirmed["baseline_internal_authorization"] is True


def test_pure_vla_runtime_rearm_without_attachments_preserves_authorization_mode() -> (
    None
):
    facade = _runtime()
    facade._controller_mode = "pi0_nav_pick_only"
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC

    preflight = facade.prepare_vla_invocation(
        invocation_id="baseline-empty",
        call_index=1,
        vla_status=None,
        baseline_internal_authorization=True,
    )
    confirmed = facade.prepare_vla_invocation(
        invocation_id="baseline-empty",
        call_index=1,
        vla_status={"actions_enabled": True},
        baseline_internal_authorization=True,
    )

    assert preflight["primitive_success"] is True
    assert preflight["attachment_count_at_invocation_start"] == 0
    assert preflight["current_object_visual_authorization"] is None
    assert confirmed["primitive_success"] is True
    assert confirmed["stop_reason"] == "vla_runtime_rearmed"
    assert confirmed["baseline_internal_authorization"] is True


def test_vla_rearm_rejects_authorization_mode_change_without_attachments() -> None:
    facade = _runtime()
    facade._controller_mode = "pi0_nav_pick_only"
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC

    facade.prepare_vla_invocation(
        invocation_id="baseline-mode-change",
        call_index=1,
        vla_status=None,
        baseline_internal_authorization=True,
    )

    with pytest.raises(RuntimeError, match="authorization mode changed"):
        facade.prepare_vla_invocation(
            invocation_id="baseline-mode-change",
            call_index=1,
            vla_status={"actions_enabled": True},
            baseline_internal_authorization=False,
        )


def test_vla_rearm_rejects_call_index_change_after_preflight() -> None:
    facade = _runtime()
    facade._controller_mode = "pi0_nav_pick_only"
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC

    facade.prepare_vla_invocation(
        invocation_id="baseline-index-change",
        call_index=1,
        vla_status=None,
        baseline_internal_authorization=True,
    )

    with pytest.raises(RuntimeError, match="call index does not match preflight"):
        facade.prepare_vla_invocation(
            invocation_id="baseline-index-change",
            call_index=2,
            vla_status={"actions_enabled": True},
            baseline_internal_authorization=True,
        )


def test_clear_active_vla_invocation_resets_all_phase_authority() -> None:
    facade = _runtime()
    facade._active_vla_invocation = "pending"
    facade._active_vla_call_index = 7
    facade._pending_vla_visual_authorization = {"source": "test"}
    facade._pending_vla_attachment_snapshot = {"available": True}
    facade._pending_vla_baseline_internal_authorization = True

    facade._clear_active_vla_invocation_state()

    assert facade._active_vla_invocation is None
    assert facade._active_vla_call_index is None
    assert facade._pending_vla_visual_authorization is None
    assert facade._pending_vla_attachment_snapshot is None
    assert facade._pending_vla_baseline_internal_authorization is False


def test_no_global_first_or_exactly_once_state_remains():
    facade = _runtime()
    assert not hasattr(facade, "_pi0_invocation_consumed")
    physical = facade.guard_tool_call(
        name="move_to",
        input_dict={
            "hand": "left",
            "visual_hand_check": _visual_hand_check("left"),
            "target": {"delta_xyz": [0, 0, 0.01], "frame": "world"},
        },
    )
    assert "pi0_nav_pick_required_first" not in physical["failed_preconditions"]


def test_dispatcher_rejects_removed_task_specific_rpc():
    dispatcher = _MainThreadDispatcher.__new__(_MainThreadDispatcher)
    dispatcher._env = _runtime()
    for name in (
        "inspect_post_pick_state",
        "prepress_move_to",
        "post_pick_direct_finger_toggle",
        "post_success_hold_frames",
    ):
        try:
            dispatcher._dispatch(f"env.{name}", (), {})
        except ValueError as exc:
            assert "unknown BEHAVIOR env RPC" in str(exc)
        else:
            raise AssertionError(f"removed RPC {name} was accepted")


def test_finish_is_not_an_env_rpc_or_facade_method():
    assert "finish" not in _ENV_RPC_METHODS
    assert not hasattr(BehaviorEnvFacade, "finish")


def test_env_shutdown_seals_without_invoking_broken_native_destructors():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    calls = []
    facade._finalize_video_segment = lambda: calls.append("video")
    facade._env = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(AssertionError("native close called"))
    )

    facade.shutdown()

    assert calls == ["video"]


def test_shutdown_receipt_is_written_after_artifact_flush(tmp_path):
    (tmp_path / "episode.mp4").write_bytes(b"video")
    (tmp_path / "behavior_action_trace.jsonl").write_text("{}\n")

    _flush_shutdown_artifacts(tmp_path)

    receipt = __import__("json").loads(
        (tmp_path / "env_shutdown_receipt.json").read_text()
    )
    assert receipt["status"] == "sealed"
    assert receipt["exit_strategy"] == "controlled_fast_exit_without_native_destructor"


def _pi0_chunk_runtime(tmp_path, step_result):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    wrapped = {
        "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["test task"],
    }
    facade._env = SimpleNamespace(
        _direct_process=SimpleNamespace(step_env=step_result),
        _wrap_obs=lambda _raw: wrapped,
    )
    facade._meta = {"max_episode_steps": 256}
    facade._env_steps = 0
    facade._done = False
    facade._last_observation = {
        "main_images": wrapped["main_images"][0],
        "wrist_images": wrapped["wrist_images"][0],
        "states": wrapped["states"][0],
        "task_descriptions": wrapped["task_descriptions"][0],
    }
    facade._last_info = {
        "done": {
            "success": False,
            "termination_conditions": {
                "timeout": {"done": False, "success": False},
                "predicate": {"done": False, "success": False},
            },
        }
    }
    facade._run_nonce = "run"
    facade._attempt_nonce = "attempt"
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
    facade._active_vla_invocation = "invocation"
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
    return facade


def _behavior_done(
    *,
    success: bool,
    predicate_done: bool,
    predicate_success: bool,
    timeout_done: bool = False,
) -> dict:
    return {
        "done": {
            "success": success,
            "termination_conditions": {
                "timeout": {"done": timeout_done, "success": False},
                "predicate": {
                    "done": predicate_done,
                    "success": predicate_success,
                },
            },
        }
    }


def test_pi0_finalize_freezes_success_at_immediate_stop_boundary(
    tmp_path,
):
    calls = {"count": 0}

    def step_result(_action, *, need_obs):
        del need_obs
        calls["count"] += 1
        success = calls["count"] == 1
        return (
            object(),
            np.array([float(success)]),
            np.array([success]),
            np.array([False]),
            [
                _behavior_done(
                    success=success,
                    predicate_done=success,
                    predicate_success=success,
                )
            ],
        )

    facade = _pi0_chunk_runtime(tmp_path, step_result)
    facade._step_action_chunk(
        np.zeros((32, 23), dtype=np.float32),
        observe_final=True,
        pi0_nav_pick=True,
    )
    assert facade._last_info["done"]["success"] is True

    finalized = facade.finalize_paused_runtime(
        {
            "actions_enabled": False,
            "healthz": {"actions_enabled": False},
        }
    )

    assert finalized["task_success"] is True
    assert finalized["official_success_source"] == 'info["done"]["success"]'
    assert finalized["official_success_receipt"]["env_step"] == 1
    assert facade._motion_frozen is True
    assert facade._controller_state == "frozen"
    assert facade._action_source == "frozen"
    assert facade._vla_actions_enabled is False
    assert facade._done is True
    assert facade._active_vla_invocation is None
    assert facade._active_vla_call_index is None
    assert facade._pending_vla_baseline_internal_authorization is False


def test_pi0_finalize_reports_boolean_false_without_raw_official_success():
    facade = _runtime()
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    facade._active_vla_invocation = "call-1"
    facade._active_vla_call_index = 1
    facade._pending_vla_baseline_internal_authorization = True
    facade._official_success_receipt = None
    facade._env_steps = 32

    def switch_controller(target, **_kwargs):
        previous = facade._controller_state
        facade._controller_state = target
        return {
            "from": previous,
            "to": target,
            "changed": target != previous,
        }

    facade._switch_controller = switch_controller

    finalized = facade.finalize_paused_runtime(
        {
            "actions_enabled": False,
            "healthz": {"actions_enabled": False, "pid": 123},
            "endpoint": "http://127.0.0.1:9999",
        }
    )

    assert finalized["lifecycle_finalized"] is True
    assert finalized["vla_actions_enabled"] is False
    assert finalized["controller_state"] == "planner"
    assert finalized["task_success"] is False
    assert type(finalized["task_success"]) is bool
    assert finalized["official_success_source"] == 'info["done"]["success"]'
    assert "official_success_receipt" not in finalized
    assert facade._official_success_latched is False
    assert facade._official_success_receipt is None
    assert facade._last_info["done"]["success"] is False
    assert facade._env_steps == 32
    assert facade._active_vla_invocation is None
    assert facade._active_vla_call_index is None
    assert facade._pending_vla_baseline_internal_authorization is False


def test_pi0_hard_failure_interrupts_chunk_without_success_receipt(tmp_path):
    calls = {"count": 0}

    def step_result(_action, *, need_obs):
        del need_obs
        calls["count"] += 1
        hard_failure = calls["count"] == 5
        return (
            object(),
            np.array([0.0]),
            np.array([hard_failure]),
            np.array([False]),
            [
                _behavior_done(
                    success=False,
                    predicate_done=hard_failure,
                    predicate_success=False,
                )
            ],
        )

    facade = _pi0_chunk_runtime(tmp_path, step_result)
    result = facade._step_action_chunk(
        np.zeros((32, 23), dtype=np.float32),
        observe_final=True,
        pi0_nav_pick=True,
    )

    assert calls["count"] == 5
    assert result[2:4] == (True, False)
    monitor = result[4]["_rpent"]["pi0_nav_pick_monitor"]
    assert monitor["executed_steps"] == 5
    assert monitor["terminal_classification"]["hard_terminated"] is True
    assert monitor["official_success_receipt"] is None
    assert facade._official_success_latched is False
    assert facade._done is True


def test_pi0_success_and_timeout_latches_success_then_hard_stops(tmp_path):
    calls = {"count": 0}

    def step_result(_action, *, need_obs):
        del need_obs
        calls["count"] += 1
        simultaneous = calls["count"] == 4
        return (
            object(),
            np.array([float(simultaneous)]),
            np.array([simultaneous]),
            np.array([simultaneous]),
            [
                _behavior_done(
                    success=simultaneous,
                    predicate_done=simultaneous,
                    predicate_success=simultaneous,
                    timeout_done=simultaneous,
                )
            ],
        )

    facade = _pi0_chunk_runtime(tmp_path, step_result)
    result = facade._step_action_chunk(
        np.zeros((32, 23), dtype=np.float32),
        observe_final=True,
        pi0_nav_pick=True,
    )

    assert calls["count"] == 4
    assert result[2:4] == (True, True)
    monitor = result[4]["_rpent"]["pi0_nav_pick_monitor"]
    assert monitor["executed_steps"] == 4
    assert monitor["terminal_classification"]["hard_truncated"] is True
    assert monitor["terminal_classification"]["soft_success_termination"] is False
    assert monitor["official_success_receipt"]["env_step"] == 4
    assert facade._official_success_latched is True
    assert facade._motion_frozen is False
    assert facade._done is True


@pytest.mark.parametrize(
    ("info", "terminated", "truncated", "expected_reason"),
    [
        (
            _behavior_done(
                success=False,
                predicate_done=True,
                predicate_success=False,
            ),
            True,
            False,
            "hard_termination",
        ),
        (
            _behavior_done(
                success=True,
                predicate_done=True,
                predicate_success=True,
                timeout_done=True,
            ),
            True,
            True,
            "hard_truncation",
        ),
        (
            {"done": {"success": True}},
            True,
            False,
            "malformed_or_inconsistent_terminal_envelope",
        ),
    ],
)
def test_pi0_hard_terminal_classification_fails_closed(
    info,
    terminated,
    truncated,
    expected_reason,
):
    classified = env_server_module._classify_pi0_terminal_step(
        info=info,
        raw_terminated=np.array([terminated]),
        raw_truncated=np.array([truncated]),
    )

    assert classified["soft_success_termination"] is False
    assert classified["hard_terminated"] or classified["hard_truncated"]
    assert classified["terminal_classification_reason"] == expected_reason


def test_non_pi0_success_still_freezes_immediately(tmp_path):
    calls = {"count": 0}

    def step_result(_action, *, need_obs):
        del need_obs
        calls["count"] += 1
        return (
            object(),
            np.array([1.0]),
            np.array([True]),
            np.array([False]),
            [
                _behavior_done(
                    success=True,
                    predicate_done=True,
                    predicate_success=True,
                )
            ],
        )

    facade = _pi0_chunk_runtime(tmp_path, step_result)
    result = facade._step_action_chunk(
        np.zeros((32, 23), dtype=np.float32),
        observe_final=True,
        pi0_nav_pick=False,
    )

    assert calls["count"] == 1
    assert result[2] is True
    assert facade._official_success_latched is True
    assert facade._motion_frozen is True
    assert facade._controller_state == "frozen"
    assert facade._done is True


def test_env_client_kwargs_dispatch_through_real_server_signatures():
    facade = _runtime()
    facade._meta = {}
    run_nonce = "1" * 32
    facade._run_nonce = run_nonce
    facade._physical_gripper_opening = lambda hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 1,
        "hands": ["left"],
        "identity_conflict": False,
        "attached_objects": {},
    }
    calls = []
    backend = SimpleNamespace(
        get_eef_pose=lambda hand: (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
    )
    planner = SimpleNamespace(
        backend=backend,
        move_to=lambda **kwargs: (
            calls.append(("move_to", kwargs))
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "moved",
                "metrics": _successful_whole_body_execution(),
            }
        ),
        press=lambda **kwargs: (
            calls.append(("press", kwargs))
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "pressed",
                "metrics": _successful_whole_body_execution(),
            }
        ),
    )
    facade._planner = planner
    facade._projection_receipts["projection-current"] = {
        "run_nonce": run_nonce,
        "attempt_nonce": "attempt",
        "env_step": 0,
        "projection_id": "projection-current",
        "world_point": [0.0, 0.0, 0.0],
        "camera_facing_normal": [0.0, 0.0, 1.0],
    }
    dispatcher = _MainThreadDispatcher(facade, threading.Event())

    class _BridgeRpc:
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            return dispatcher._dispatch(method, args, kwargs or {})

    client = BehaviorEnvClient(_BridgeRpc(), expected_meta={})
    client.move_to(
        hand="left",
        target={"delta_xyz": [0.0, 0.0, 0.05], "frame": "world"},
        visual_hand_check=_visual_hand_check("left"),
        position_tolerance_m=0.01,
        max_travel_m=0.10,
        timeout_s=30.0,
    )
    client.move_to(
        hand="left",
        target={"projection_id": "projection-current", "standoff_m": 0.0},
        visual_hand_check=_visual_hand_check("left"),
        position_tolerance_m=0.01,
        max_travel_m=0.10,
        timeout_s=30.0,
    )
    assert "projection-current" not in facade._consumed_projection_receipts
    client.press(
        hand="right",
        projection_id="projection-current",
        travel_m=0.02,
        visual_hand_check=_visual_hand_check("right"),
        timeout_s=30.0,
    )

    assert calls[0][0] == "move_to"
    assert calls[0][1]["hand"] == "left"
    assert np.allclose(calls[0][1]["target_xyz"], [0.0, 0.0, 0.05])
    assert calls[0][1]["position_tolerance_m"] == 0.01
    assert calls[0][1]["timeout_s"] == 30.0
    assert "allow_trunk_assist" not in calls[0][1]
    assert calls[1][0] == "move_to"
    assert "allow_trunk_assist" not in calls[1][1]
    assert calls[2][0] == "press"
    assert calls[2][1]["hand"] == "right"
    assert calls[2][1]["travel_m"] == 0.02
    assert "approach_distance_m" not in calls[2][1]
    assert "press_depth_m" not in calls[2][1]
    assert "projection-current" in facade._consumed_projection_receipts


def _rotate_runtime(*, held_hand: str | None) -> tuple[BehaviorEnvFacade, list]:
    facade = _runtime()
    facade._run_nonce = "run"
    facade._public_observed_frame_ids = {"head:0:visual-selection"}
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: SimpleNamespace(
            frame_id=frame_id,
            step_index=0,
            capture_group_id="capture:0:visual-selection",
        )
    )
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": int(held_hand is not None),
        "hands": [] if held_hand is None else [held_hand],
        "identity_conflict": False,
        "attached_objects": {},
    }
    calls: list = []
    facade._planner = SimpleNamespace(
        rotate_wrist=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "reached",
                "metrics": _successful_whole_body_execution(),
            }
        ),
        warmup=lambda: {"status": "complete"},
    )
    return facade, calls


def _visual_hand_check(hand: str) -> dict[str, str]:
    return {
        "camera": "head",
        "frame_id": "head:0:visual-selection",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _navigation_visual_check(
    *,
    frame_id: str = "head:0:visual-selection",
    camera: str = "head",
    assessment: str = "navigation_target_visually_confirmed",
) -> dict[str, str]:
    return {
        "camera": camera,
        "frame_id": frame_id,
        "assessment": assessment,
    }


def _install_navigation_projection(
    facade: BehaviorEnvFacade,
    *,
    projection_id: str = "projection-navigation",
    frame_id: str = "head:0:visual-selection",
    capture_group_id: str = "capture:0:visual-selection",
    env_step: int = 0,
) -> str:
    facade._projection_receipts[projection_id] = {
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "env_step": env_step,
        "camera": "head",
        "resolved_camera": "head",
        "frame_id": frame_id,
        "capture_group_id": capture_group_id,
        "projection_id": projection_id,
        "world_point": [1.5, -0.4, 0.0],
        "camera_facing_normal": [1.0, 0.0, 0.0],
        "confidence": 0.95,
    }
    return projection_id


def _successful_navigation_isolation() -> dict[str, object]:
    checks = dict.fromkeys(
        (
            "base_z_locked",
            "base_roll_pitch_locked",
            "trunk_locked",
            "left_arm_locked",
            "right_arm_locked",
            "left_gripper_command_locked",
            "right_gripper_command_locked",
            "left_attachment_identity_unchanged",
            "right_attachment_identity_unchanged",
        ),
        True,
    )
    return {
        "available": True,
        "ok": True,
        "mode": "base_only",
        "checks_performed": 1,
        "checks": checks,
        "max_observed": {
            "base_z_drift_m": 0.0,
            "base_roll_pitch_drift_rad": 0.0,
            "trunk_drift_rad": 0.0,
            "left_arm_drift_rad": 0.0,
            "right_arm_drift_rad": 0.0,
            "left_gripper_command_drift": 0.0,
            "right_gripper_command_drift": 0.0,
        },
        "thresholds": {
            "base_z_m": 0.01,
            "base_roll_pitch_rad": np.deg2rad(1.0),
            "articulation_rad": 0.01,
            "gripper_command": 1e-6,
        },
    }


def _successful_whole_body_execution() -> dict[str, object]:
    return {
        "motion_scope": "whole_body",
        "whole_body_execution": {
            "available": True,
            "ok": True,
            "collision_certificate_verified_before_first_action": True,
            "dual_attachment_checked_each_nonterminal_step": True,
            "raw_success_checked_after_each_action": True,
            "raw_success_preempts_post_step_safety_checks": True,
        },
    }


def _successful_gripper_isolation(hand: str) -> dict[str, object]:
    return {
        "single_arm_isolation": {
            "available": True,
            "ok": True,
            "selected_hand": hand,
            "mode": "gripper_only",
            "stop_reason": "single_arm_isolation_verified",
        }
    }


def _fresh_visual_hand_check(
    facade: BehaviorEnvFacade,
    hand: str,
) -> dict[str, str]:
    frame_id = f"head:{facade._env_steps}:visual-selection"
    frame = SimpleNamespace(
        frame_id=frame_id,
        step_index=int(facade._env_steps),
        capture_group_id=f"capture:{facade._env_steps}:visual-selection",
    )
    facade._public_observed_frame_ids.add(frame_id)
    facade._latest_public_head_frame_id = frame_id
    previous_get_current = facade._frame_cache.get_current
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, requested_id: (
            frame
            if camera == "head" and requested_id == frame_id
            else previous_get_current(camera, requested_id)
        )
    )
    return {
        "camera": "head",
        "frame_id": frame_id,
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def test_navigate_to_revalidates_fresh_same_frame_projection_and_preserves_raw_success():
    facade = _runtime()
    projection_id = _install_navigation_projection(facade)
    planner_calls: list[dict] = []

    def navigate(**kwargs):
        planner_calls.append(kwargs)
        facade._env_steps += 3
        return {
            "primitive_success": True,
            # Runtime must not trust a planner-level success claim.
            "task_success": True,
            "stop_reason": "navigation_complete",
            "metrics": {
                "navigation_isolation": _successful_navigation_isolation(),
                "navigation_path": {
                    "source": "official_robot_eroded_traversability",
                    "full_path_used": False,
                },
                "base_goal": [1.0, -0.2, 0.1],
            },
        }

    facade._planner = SimpleNamespace(navigate_to=navigate)

    result = facade.navigate_to(
        projection_id=projection_id,
        navigation_visual_check=_navigation_visual_check(),
        standoff_m=0.9,
        max_travel_m=1.2,
        timeout_s=250.0,
    )

    assert len(planner_calls) == 1
    assert np.allclose(planner_calls[0].pop("target_xyz"), [1.5, -0.4, 0.0])
    assert planner_calls[0] == {
        "standoff_m": 0.9,
        "max_travel_m": 1.2,
        "timeout_s": 250.0,
    }
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["official_success_source"] == 'info["done"]["success"]'
    assert result["requested_projection_id"] == projection_id
    assert result["navigation_visual_evidence"]["frame_id"] == (
        "head:0:visual-selection"
    )
    assert result["navigation_isolation"]["mode"] == "base_only"
    assert result["navigation_isolation"]["ok"] is True
    assert result["attachment_isolation"]["passed"] is True
    assert projection_id in facade._consumed_projection_receipts
    assert "navigation_path" not in repr(result)
    assert "base_goal" not in repr(result)
    assert "world_point" not in repr(result)


@pytest.mark.parametrize(
    "relative_motion",
    [
        {"kind": "translation", "direction": "forward", "distance_m": 0.25},
        {"kind": "translation", "direction": "backward", "distance_m": 0.25},
        {"kind": "rotation", "direction": "left", "angle_deg": 90.0},
        {"kind": "rotation", "direction": "right", "angle_deg": 90.0},
    ],
)
def test_navigate_to_relative_motion_needs_no_projection_and_preserves_isolation(
    relative_motion,
):
    facade = _runtime()
    planner_calls: list[dict] = []

    def navigate(**kwargs):
        planner_calls.append(kwargs)
        facade._env_steps += 1
        return {
            "primitive_success": True,
            "stop_reason": "reached",
            "recoverable": True,
            "metrics": {
                "navigation_isolation": _successful_navigation_isolation(),
            },
        }

    facade._planner = SimpleNamespace(navigate_to=navigate)

    result = facade.navigate_to(
        relative_motion=relative_motion,
        timeout_s=120.0,
    )

    assert planner_calls == [
        {
            "relative_motion": relative_motion,
            "timeout_s": 120.0,
        }
    ]
    assert result["primitive_success"] is True
    assert result["requested_relative_motion"] == relative_motion
    assert "requested_projection_id" not in result
    assert "navigation_visual_evidence" not in result
    assert facade._consumed_projection_receipts == set()
    assert result["navigation_isolation"]["ok"] is True
    assert result["attachment_isolation"]["passed"] is True


def test_navigate_to_relative_motion_rejects_projection_arguments_before_switch():
    facade = _runtime()
    planner_calls: list[dict] = []
    switch_calls: list[str] = []
    facade._planner = SimpleNamespace(
        navigate_to=lambda **kwargs: planner_calls.append(kwargs)
    )
    facade._switch_controller = lambda target: switch_calls.append(target)

    with pytest.raises(ValueError, match="mutually exclusive"):
        facade.navigate_to(
            projection_id="projection-current",
            relative_motion={
                "kind": "translation",
                "direction": "forward",
                "distance_m": 0.25,
            },
        )

    assert planner_calls == []
    assert switch_calls == []
    assert facade._env_steps == 0


@pytest.mark.parametrize("mode", ["projection", "relative"])
def test_navigate_to_raw_success_preempts_post_step_isolation_downgrade(mode):
    facade = _runtime()
    planner_calls: list[dict] = []

    def navigate(**kwargs):
        planner_calls.append(kwargs)
        facade._env_steps += 1
        facade._official_success_latched = True
        facade._official_success_receipt = {
            "source": 'info["done"]["success"]',
            "env_step": 1,
        }
        facade._last_info = {"done": {"success": True}}
        return {
            "primitive_success": True,
            "stop_reason": "official_task_success",
            "recoverable": False,
            "metrics": {},
        }

    facade._planner = SimpleNamespace(navigate_to=navigate)
    if mode == "projection":
        projection_id = _install_navigation_projection(facade)
        kwargs = {
            "projection_id": projection_id,
            "navigation_visual_check": _navigation_visual_check(),
        }
    else:
        kwargs = {
            "relative_motion": {
                "kind": "rotation",
                "direction": "left",
                "angle_deg": 90.0,
            }
        }

    result = facade.navigate_to(**kwargs)

    assert len(planner_calls) == 1
    assert facade._env_steps == 1
    assert result["primitive_success"] is True
    assert result["task_success"] is True
    assert result["stop_reason"] == "official_task_success"
    assert result["_finish"] is True
    assert result["navigation_isolation"]["available"] is False


@pytest.mark.parametrize("stop_reason", ["timeout", "error"])
def test_navigate_to_preserves_no_action_planning_failure_without_isolation(
    stop_reason,
):
    facade = _runtime()
    projection_id = _install_navigation_projection(facade)
    facade._planner = SimpleNamespace(
        navigate_to=lambda **_kwargs: {
            "primitive_success": False,
            "stop_reason": stop_reason,
            "recoverable": True,
            "metrics": {},
        }
    )

    result = facade.navigate_to(
        projection_id=projection_id,
        navigation_visual_check=_navigation_visual_check(),
        standoff_m=0.9,
        max_travel_m=1.2,
        timeout_s=250.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == stop_reason
    assert result["navigation_isolation"]["available"] is False
    assert result["attachment_isolation"]["passed"] is True
    assert projection_id not in facade._consumed_projection_receipts


def test_navigate_to_missing_isolation_after_action_remains_fail_closed():
    facade = _runtime()
    projection_id = _install_navigation_projection(facade)

    def navigate(**_kwargs):
        facade._env_steps += 1
        return {
            "primitive_success": False,
            "stop_reason": "timeout",
            "recoverable": True,
            "metrics": {},
        }

    facade._planner = SimpleNamespace(navigate_to=navigate)

    result = facade.navigate_to(
        projection_id=projection_id,
        navigation_visual_check=_navigation_visual_check(),
        standoff_m=0.9,
        max_travel_m=1.2,
        timeout_s=250.0,
    )

    assert result["primitive_success"] is False
    assert result["stop_reason"] == "navigation_isolation_feedback_unavailable"
    assert result["navigation_isolation"]["available"] is False
    assert result["attachment_isolation"]["passed"] is True
    assert projection_id in facade._consumed_projection_receipts


@pytest.mark.parametrize(
    "case",
    [
        "missing_projection",
        "consumed_projection",
        "wrong_camera",
        "wrong_assessment",
        "unpublished_frame",
        "nonlatest_frame",
        "stale_env_step",
        "different_projection_frame",
        "attachment_feedback_unavailable",
    ],
)
def test_navigate_to_fresh_same_frame_negative_matrix_is_zero_side_effect(case):
    facade = _runtime()
    projection_id = _install_navigation_projection(facade)
    visual_check = _navigation_visual_check()
    planner_calls: list[dict] = []
    switch_calls: list[str] = []
    facade._planner = SimpleNamespace(
        navigate_to=lambda **kwargs: planner_calls.append(kwargs)
    )
    facade._switch_controller = lambda target: (
        switch_calls.append(target) or {"changed": True}
    )

    if case == "missing_projection":
        facade._projection_receipts.clear()
    elif case == "consumed_projection":
        facade._consumed_projection_receipts.add(projection_id)
    elif case == "wrong_camera":
        visual_check = _navigation_visual_check(camera="left_wrist")
    elif case == "wrong_assessment":
        visual_check = _navigation_visual_check(
            assessment="target_point_visually_confirmed"
        )
    elif case == "unpublished_frame":
        facade._public_observed_frame_ids.clear()
    elif case == "nonlatest_frame":
        facade._latest_public_head_frame_id = "head:0:newer"
    elif case == "stale_env_step":
        facade._env_steps = 1
    elif case == "different_projection_frame":
        facade._projection_receipts[projection_id]["frame_id"] = (
            "head:0:different-projection"
        )
    elif case == "attachment_feedback_unavailable":
        facade._attachment_runtime_facts = lambda: _attachment_facts(available=False)
    else:  # pragma: no cover - guards the closed parameter matrix
        raise AssertionError(case)

    before = (
        facade._env_steps,
        facade._controller_state,
        set(facade._consumed_projection_receipts),
    )
    with pytest.raises((RuntimeError, ValueError)):
        facade.navigate_to(
            projection_id=projection_id,
            navigation_visual_check=visual_check,
            standoff_m=0.85,
            max_travel_m=1.0,
            timeout_s=300.0,
        )

    assert planner_calls == []
    assert switch_calls == []
    assert (
        facade._env_steps,
        facade._controller_state,
        set(facade._consumed_projection_receipts),
    ) == before


def test_rotate_wrist_requires_visual_confirmation_of_explicit_physical_hand():
    for first_hand, second_hand in (("left", "right"), ("right", "left")):
        facade, calls = _rotate_runtime(held_hand=None)

        first = facade.rotate_wrist(
            hand=first_hand,
            relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
            visual_hand_check=_visual_hand_check(first_hand),
        )
        second = facade.rotate_wrist(
            hand=second_hand,
            relative_axis_angle=[0.0, 1.0, 0.0, -0.2],
            visual_hand_check=_visual_hand_check(second_hand),
        )

        assert [call["hand"] for call in calls] == [first_hand, second_hand]
        assert [call["timeout_s"] for call in calls] == [
            ROTATE_WRIST_RUNTIME_TIMEOUT_S,
            ROTATE_WRIST_RUNTIME_TIMEOUT_S,
        ]
        assert first["requested_hand"] == first_hand
        assert first["resolved_hand"] == first_hand
        assert first["visual_hand_evidence"]["selected_hand"] == first_hand
        assert first["metrics"]["runtime_timeout_s"] == ROTATE_WRIST_RUNTIME_TIMEOUT_S
        assert second["requested_hand"] == second_hand
        assert second["resolved_hand"] == second_hand
        assert second["visual_hand_evidence"]["selected_hand"] == second_hand
        assert second["metrics"]["runtime_timeout_s"] == ROTATE_WRIST_RUNTIME_TIMEOUT_S
        assert "attached_rotate_receipt" not in second


def test_rotate_wrist_runtime_does_not_accept_a_caller_timeout():
    parameters = inspect.signature(BehaviorEnvFacade.rotate_wrist).parameters
    assert "timeout_s" not in parameters
    assert "target_quat_xyzw" not in parameters


def test_rotate_wrist_literal_hand_requires_fresh_public_head_visual_selection():
    facade, calls = _rotate_runtime(held_hand=None)

    result = facade.rotate_wrist(
        hand="right",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_visual_hand_check("right"),
    )

    assert calls[0]["hand"] == "right"
    assert result["requested_hand"] == "right"
    assert result["resolved_hand"] == "right"
    assert result["hand_selection_source"] == "llm_visual_hand_selection"
    assert "attached_rotate_receipt" not in result
    assert (
        result["visual_hand_evidence"].items()
        >= {
            "schema_version": 1,
            "source": "llm_fresh_public_head_observation",
            "run_nonce": "run",
            "attempt_nonce": "attempt",
            "attempt_index": 1,
            "env_step": 0,
            "camera": "head",
            "resolved_camera": "head",
            "frame_id": "head:0:visual-selection",
            "capture_group_id": "capture:0:visual-selection",
            "selected_hand": "right",
            "assessment": "selected_hand_visually_confirmed",
        }.items()
    )
    assert result["visual_hand_evidence"]["kind"] == "visual_hand_authorization"
    assert len(result["visual_hand_evidence"]["receipt_sha256"]) == 64


def test_rotate_wrist_explicit_hand_allows_dual_attachment_controller_switch():
    facade, calls = _rotate_runtime(held_hand=None)
    facade._controller_state = _CONTROLLER_VLA
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/left_object")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/right_object")},
    )
    switch_calls: list[str] = []

    def switch(target: str) -> dict:
        switch_calls.append(target)
        facade._controller_state = target
        return {"from": _CONTROLLER_VLA, "to": target, "changed": True}

    facade._switch_controller = switch

    result = facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_visual_hand_check("left"),
    )

    assert result["primitive_success"] is True
    assert calls[0]["hand"] == "left"
    assert switch_calls == ["planner"]


def test_controller_switch_preserves_dual_attachment_fingerprints():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._controller_state = _CONTROLLER_VLA
    facade._motion_in_flight = False
    facade._official_success_latched = False
    facade._base_controller_mode = "velocity"
    facade._action_source = "pi0_vla"
    facade._vla_actions_enabled = True
    facade._robot = lambda: SimpleNamespace(reload_controllers=lambda _config: None)
    facade._reload_base_controller_position = lambda: {
        "from": "velocity",
        "to": "position",
    }
    facade._planner = SimpleNamespace(on_runtime_state_changed=lambda: None)
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/left_object")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/right_object")},
    )

    result = facade._switch_controller("planner")

    assert result["changed"] is True
    assert facade._controller_state == "planner"
    assert facade._action_source == "planner"
    assert facade._vla_actions_enabled is False


def test_rotate_wrist_literal_hand_rejects_missing_or_mismatched_visual_selection():
    facade, calls = _rotate_runtime(held_hand=None)

    for visual_check in (
        None,
        _visual_hand_check("left"),
        {
            **_visual_hand_check("right"),
            "camera": "left_wrist",
        },
        {
            **_visual_hand_check("right"),
            "assessment": "hand_probably_visible",
        },
    ):
        try:
            facade.rotate_wrist(
                hand="right",
                relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
                visual_hand_check=visual_check,
            )
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid literal-hand visual selection was accepted")

    assert calls == []


def test_rotate_wrist_guard_validates_literal_hand_visual_selection():
    facade, _calls = _rotate_runtime(held_hand=None)

    rejected = facade.guard_tool_call(
        name="rotate_wrist",
        input_dict={
            "hand": "left",
            "relative_axis_angle": [0.0, 1.0, 0.0, 0.2],
        },
    )
    assert "visual_hand_selection_unavailable" in rejected["failed_preconditions"]

    allowed = facade.guard_tool_call(
        name="rotate_wrist",
        input_dict={
            "hand": "left",
            "relative_axis_angle": [0.0, 1.0, 0.0, 0.2],
            "visual_hand_check": _visual_hand_check("left"),
        },
    )
    assert allowed["failed_preconditions"] == []


def test_rotate_wrist_literal_hand_rejects_stale_or_unpublished_frame():
    facade, calls = _rotate_runtime(held_hand=None)
    visual_check = _visual_hand_check("left")

    facade._public_observed_frame_ids.clear()
    try:
        facade.rotate_wrist(
            hand="left",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
            visual_hand_check=visual_check,
        )
    except RuntimeError as exc:
        assert "public observe" in str(exc)
    else:
        raise AssertionError("unpublished frame was accepted")

    facade._public_observed_frame_ids.add(visual_check["frame_id"])
    facade._env_steps = 1
    try:
        facade.rotate_wrist(
            hand="left",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
            visual_hand_check=visual_check,
        )
    except RuntimeError as exc:
        assert "current env step" in str(exc)
    else:
        raise AssertionError("stale frame was accepted")

    assert calls == []


def test_dual_attachment_allows_explicit_hands_but_press_requires_free_hand():
    facade = _runtime()
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/left_object")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/right_object")},
    )

    for name in ("move_to", "rotate_wrist", "close", "open"):
        guarded = facade.guard_tool_call(
            name=name,
            input_dict={
                "hand": "left",
                "visual_hand_check": _visual_hand_check("left"),
            },
        )
        assert guarded["failed_preconditions"] == []

    facade._physical_gripper_opening = lambda _hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    press = facade.guard_tool_call(
        name="press",
        input_dict={
            "hand": "left",
            "visual_hand_check": _visual_hand_check("left"),
        },
    )
    assert "press_hand_must_be_attachment_free" in press["failed_preconditions"]


def test_same_attachment_identity_under_both_hands_fails_closed():
    facade = _runtime()
    shared_root = SimpleNamespace(prim_path="/World/shared_object")
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_attached_object=lambda _hand: {
                "eef_link": shared_root,
            }
        )
    )
    facade._attachment_runtime_facts = (
        BehaviorEnvFacade._attachment_runtime_facts.__get__(
            facade,
            BehaviorEnvFacade,
        )
    )

    facts = facade._attachment_runtime_facts()

    assert facts["available"] is False
    assert facts["identity_conflict"] is True
    assert facts["attachment_count"] == 2
    rejected = facade.guard_tool_call(
        name="move_to",
        input_dict={
            "hand": "left",
            "visual_hand_check": _visual_hand_check("left"),
        },
    )
    assert rejected["failed_preconditions"] == ["visual_hand_selection_unavailable"]


def test_public_capability_reports_per_hand_attachment_facts_without_identity():
    facade = _runtime()
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left={"left_eef_link": SimpleNamespace(prim_path="/World/private_can")},
        right={"right_eef_link": SimpleNamespace(prim_path="/World/private_bin")},
    )
    facade._physical_gripper_opening = lambda _hand: 0.02

    capability = facade._sanitized_capability_summary()

    assert capability["attachments"] == {
        "available": True,
        "count": 2,
        "conflict": False,
        "by_hand": {
            "left": {"attached": True},
            "right": {"attached": True},
        },
    }
    assert capability["gripper_state"] == {"left": "open", "right": "open"}
    public_text = repr(capability)
    assert "/World/" not in public_text
    assert "fingerprint" not in public_text
    assert "held_object" not in public_text
    assert "dynamic_role" not in public_text


def test_trash_open_attached_hand_requires_exact_fresh_release_visual_check():
    facade = _runtime()
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    roots = {
        "left": {"left_eef_link": SimpleNamespace(prim_path="/World/can")},
    }
    facade._attachment_runtime_facts = lambda: _attachment_facts(
        left=roots.get("left"),
    )
    visual = _visual_hand_check("left")
    release = {
        "camera": "head",
        "frame_id": visual["frame_id"],
        "selected_hand": "left",
        "assessment": "attached_object_fully_inside_receptacle_opening",
    }

    missing = facade.guard_tool_call(
        name="open",
        input_dict={"hand": "left", "visual_hand_check": visual},
    )
    assert missing["failed_preconditions"] == ["fresh_release_visual_check_required"]
    for mutation in (
        {"camera": "left_wrist"},
        {"selected_hand": "right"},
        {"assessment": "object_near_receptacle"},
        {"frame_id": "head:0:stale"},
    ):
        invalid = facade.guard_tool_call(
            name="open",
            input_dict={
                "hand": "left",
                "visual_hand_check": visual,
                "release_visual_check": {**release, **mutation},
            },
        )
        assert "fresh_release_visual_check_required" in invalid["failed_preconditions"]
    allowed = facade.guard_tool_call(
        name="open",
        input_dict={
            "hand": "left",
            "visual_hand_check": visual,
            "release_visual_check": release,
        },
    )
    assert allowed["failed_preconditions"] == []

    def open_selected(selected_hand, **_kwargs):
        roots.pop(selected_hand)
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "opened",
            "metrics": _successful_gripper_isolation(selected_hand),
        }

    facade._planner = SimpleNamespace(_gripper_command=open_selected)
    result = facade.open(
        hand="left",
        visual_hand_check=visual,
        release_visual_check=release,
    )

    assert result["primitive_success"] is True
    assert result["requested_hand"] == "left"
    assert result["resolved_hand"] == "left"
    evidence = result["release_visual_evidence"]
    assert evidence["kind"] == "trash_release_visual_authorization"
    assert evidence["selected_hand"] == "left"
    assert evidence["semantic_target_verified"] is False
    assert evidence["collision_authorization"] is False
    assert evidence["distance_authorization"] is False
    assert "/World/" not in repr(result)


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize(
    ("name", "invoke"),
    [
        (
            "move_to",
            lambda facade, hand, check: facade.move_to(
                hand=hand,
                target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
                visual_hand_check=check,
            ),
        ),
        (
            "rotate_wrist",
            lambda facade, hand, check: facade.rotate_wrist(
                hand=hand,
                relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
                visual_hand_check=check,
            ),
        ),
        (
            "close",
            lambda facade, hand, check: facade.close(
                hand=hand,
                visual_hand_check=check,
            ),
        ),
        (
            "open",
            lambda facade, hand, check: facade.open(
                hand=hand,
                visual_hand_check=check,
            ),
        ),
        (
            "press",
            lambda facade, hand, check: facade.press(
                hand=hand,
                projection_id="projection-current",
                travel_m=0.02,
                visual_hand_check=check,
            ),
        ),
    ],
)
def test_literal_analytic_primitives_route_to_exact_selected_target(
    name,
    invoke,
    hand,
):
    facade = _runtime()
    calls: list[tuple[str, str]] = []

    def result(method: str, selected_hand: str, *, gripper_only: bool = False):
        calls.append((method, selected_hand))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "reached",
            "metrics": (
                _successful_gripper_isolation(selected_hand)
                if gripper_only
                else _successful_whole_body_execution()
            ),
        }

    facade._attachment_runtime_facts = _attachment_facts
    facade._physical_gripper_opening = lambda _hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    facade._projection_receipts["projection-current"] = {
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "env_step": 0,
        "projection_id": "projection-current",
        "world_point": [0.0, 0.0, 0.0],
        "camera_facing_normal": [0.0, 0.0, 1.0],
    }
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_eef_pose=lambda _hand: (
                np.zeros(3, dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
        ),
        move_to=lambda **kwargs: result("move_to", kwargs["hand"]),
        rotate_wrist=lambda **kwargs: result("rotate_wrist", kwargs["hand"]),
        _gripper_command=lambda selected_hand, **_kwargs: result(
            name,
            selected_hand,
            gripper_only=True,
        ),
        press=lambda **kwargs: result("press", kwargs["hand"]),
    )

    actual = invoke(facade, hand, _visual_hand_check(hand))

    assert calls == [(name, hand)]
    assert actual["primitive_success"] is True
    assert actual["requested_hand"] == hand
    assert actual["resolved_hand"] == hand
    assert "requested_role" not in actual
    assert "semantic_role" not in actual
    assert actual["visual_hand_evidence"]["selected_hand"] == hand
    if name in {"move_to", "rotate_wrist", "press"}:
        assert actual["single_arm_isolation"] is None
        assert actual["whole_body_execution"]["ok"] is True
        assert actual["metrics"]["motion_scope"] == "whole_body"
    else:
        assert actual["whole_body_execution"] is None
        assert actual["single_arm_isolation"]["selected_hand"] == hand
        assert actual["single_arm_isolation"]["ok"] is True
    assert "/World/" not in repr(actual)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_explicit_hand_requires_matching_visual_selection(hand):
    facade = _runtime()
    resolved, _source, evidence = facade._authorize_analytic_hand(
        hand,
        _visual_hand_check(hand),
    )
    assert resolved == hand
    assert evidence["selected_hand"] == hand
    other = "right" if hand == "left" else "left"
    with pytest.raises(ValueError, match="resolved physical hand"):
        facade._authorize_analytic_hand(hand, _visual_hand_check(other))


def test_visual_hand_check_rejects_same_step_nonlatest_or_expired_frame_without_side_effect():
    facade, calls = _rotate_runtime(held_hand=None)
    old_check = _visual_hand_check("left")
    facade._latest_public_head_frame_id = "head:0:newer"
    before = (
        facade._controller_state,
        facade._env_steps,
        set(facade._consumed_projection_receipts),
    )
    with pytest.raises(RuntimeError, match="latest public head"):
        facade.rotate_wrist(
            hand="left",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
            visual_hand_check=old_check,
        )
    assert calls == []
    assert (
        facade._controller_state,
        facade._env_steps,
        set(facade._consumed_projection_receipts),
    ) == before

    facade._latest_public_head_frame_id = old_check["frame_id"]
    facade._frame_cache = SimpleNamespace(
        get_current=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("frame TTL expired")
        )
    )
    with pytest.raises(RuntimeError, match="TTL expired"):
        facade.rotate_wrist(
            hand="left",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
            visual_hand_check=old_check,
        )
    assert calls == []


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("name", ["move_to", "rotate_wrist", "close", "open"])
def test_dual_attachment_explicit_execution_preserves_attachment_identity(
    name,
    hand,
):
    other = "right" if hand == "left" else "left"
    roots = {
        "left": {"left_eef_link": SimpleNamespace(prim_path="/World/left_object")},
        "right": {"right_eef_link": SimpleNamespace(prim_path="/World/right_object")},
    }
    facade = _runtime()
    switch_calls: list[str] = []
    planner_calls: list[tuple[str, str]] = []

    def facts():
        return _attachment_facts(
            left=roots.get("left"),
            right=roots.get("right"),
        )

    def planner_result(method: str, selected_hand: str, *, gripper=False):
        planner_calls.append((method, selected_hand))
        if method == "open":
            roots.pop(selected_hand)
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "reached",
            "metrics": (
                _successful_gripper_isolation(selected_hand)
                if gripper
                else _successful_whole_body_execution()
            ),
        }

    facade._attachment_runtime_facts = facts
    facade._switch_controller = lambda target: (
        switch_calls.append(target) or {"changed": True}
    )
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_eef_pose=lambda _hand: (
                np.zeros(3, dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
        ),
        move_to=lambda **kwargs: planner_result("move_to", kwargs["hand"]),
        rotate_wrist=lambda **kwargs: planner_result(
            "rotate_wrist",
            kwargs["hand"],
        ),
        _gripper_command=lambda selected_hand, **_kwargs: planner_result(
            name,
            selected_hand,
            gripper=True,
        ),
    )
    check = _visual_hand_check(hand)
    if name == "move_to":
        result = facade.move_to(
            hand=hand,
            target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
            visual_hand_check=check,
        )
    elif name == "rotate_wrist":
        result = facade.rotate_wrist(
            hand=hand,
            relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
            visual_hand_check=check,
        )
    else:
        result = getattr(facade, name)(
            hand=hand,
            visual_hand_check=check,
        )

    assert planner_calls == [(name, hand)]
    assert switch_calls == ["planner"]
    assert result["primitive_success"] is True
    if name in {"move_to", "rotate_wrist"}:
        assert result["whole_body_execution"]["ok"] is True
        assert result["single_arm_isolation"] is None
    else:
        assert result["whole_body_execution"] is None
        assert result["single_arm_isolation"]["mode"] == "gripper_only"
    attachment = result["metrics"]["attachment_isolation"]
    assert attachment["checks"]["inactive_attachment_unchanged"] is True
    assert attachment["inactive_hand"] == other


@pytest.mark.parametrize(
    ("primitive", "before_hands", "before_fp", "after_hands", "after_fp"),
    [
        ("move_to", ["left"], {"left": "a"}, ["left"], {"left": "changed"}),
        ("rotate_wrist", ["left"], {"left": "a"}, [], {}),
        ("press", [], {}, ["left"], {"left": "new"}),
        (
            "close",
            ["left"],
            {"left": "a"},
            ["left"],
            {"left": "replacement"},
        ),
        (
            "open",
            ["left", "right"],
            {"left": "a", "right": "b"},
            [],
            {},
        ),
    ],
)
def test_attachment_postconditions_fail_closed_on_identity_or_inactive_drift(
    primitive,
    before_hands,
    before_fp,
    after_hands,
    after_fp,
):
    facade = _runtime()
    before = {
        "available": True,
        "hands": before_hands,
        "fingerprints": {"left": None, "right": None, **before_fp},
        "env_step": 0,
    }
    after = {
        "available": True,
        "hands": after_hands,
        "fingerprints": {"left": None, "right": None, **after_fp},
        "env_step": 1,
    }
    facade._attachment_fingerprint_snapshot = lambda: after

    receipt = facade._attachment_postcondition_receipt(
        primitive=primitive,
        selected_hand="left",
        before=before,
    )

    assert receipt["passed"] is False
    assert all("/World/" not in str(value) for value in receipt.values())


@pytest.mark.parametrize(
    "name",
    ["move_to", "rotate_wrist", "close", "open", "press", "navigate_to"],
)
def test_controller_switch_attachment_drift_rejects_before_first_planner_action(name):
    roots = {
        "left": {"left_eef_link": SimpleNamespace(prim_path="/World/left_object")},
        "right": {"right_eef_link": SimpleNamespace(prim_path="/World/right_object")},
    }
    if name == "press":
        roots.pop("left")
    facade = _runtime()
    planner_calls: list[str] = []

    def facts():
        return _attachment_facts(
            left=roots.get("left"),
            right=roots.get("right"),
        )

    def switch(_target, **_kwargs):
        roots["right"] = {
            "right_eef_link": SimpleNamespace(prim_path="/World/replaced_object")
        }
        return {"changed": True}

    facade._attachment_runtime_facts = facts
    facade._switch_controller = switch
    facade._physical_gripper_opening = lambda _hand: 0.0
    facade._gripper_latch = {"left": -1.0, "right": -1.0}
    facade._projection_receipts["projection-current"] = {
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "env_step": 0,
        "projection_id": "projection-current",
        "world_point": [0.0, 0.0, 0.0],
        "camera_facing_normal": [0.0, 0.0, 1.0],
    }
    _install_navigation_projection(facade)
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            get_eef_pose=lambda _hand: (
                np.zeros(3, dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
        ),
        move_to=lambda **_kwargs: planner_calls.append("move_to"),
        rotate_wrist=lambda **_kwargs: planner_calls.append("rotate_wrist"),
        _gripper_command=lambda *_args, **_kwargs: planner_calls.append(name),
        press=lambda **_kwargs: planner_calls.append("press"),
        navigate_to=lambda **_kwargs: planner_calls.append("navigate_to"),
    )
    check = _visual_hand_check("left")

    with pytest.raises(RuntimeError, match="attachment identity changed"):
        if name == "move_to":
            facade.move_to(
                hand="left",
                target={"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
                visual_hand_check=check,
            )
        elif name == "rotate_wrist":
            facade.rotate_wrist(
                hand="left",
                relative_axis_angle=[0.0, 1.0, 0.0, 0.1],
                visual_hand_check=check,
            )
        elif name == "press":
            facade.press(
                hand="left",
                projection_id="projection-current",
                travel_m=0.02,
                visual_hand_check=check,
            )
        elif name == "navigate_to":
            facade.navigate_to(
                projection_id="projection-navigation",
                navigation_visual_check=_navigation_visual_check(),
                standoff_m=0.85,
                max_travel_m=1.0,
                timeout_s=300.0,
            )
        else:
            getattr(facade, name)(
                hand="left",
                visual_hand_check=check,
            )

    assert planner_calls == []


def _opposite_surface_runtime(
    tmp_path,
    *,
    held_hand: str,
) -> tuple[BehaviorEnvFacade, dict[str, object]]:
    facade, _calls = _rotate_runtime(held_hand=held_hand)
    facade._output_dir = tmp_path
    facade._pi0_nav_pick_disable_receipt_path = (
        tmp_path / "pi0_nav_pick_disable_receipt.json"
    )
    attachment_state: dict[str, object] = {
        "root": SimpleNamespace(prim_path="/World/radio"),
    }

    def attachment_facts():
        root = attachment_state["root"]
        roots = {f"{held_hand}_eef_link": root}
        return _attachment_facts(**{held_hand: roots})

    facade._attachment_runtime_facts = attachment_facts
    facade._public_observed_frame_ids = set()
    facade._current_object_visual_authorization = lambda _check, *, invocation_id: {
        "invocation_id": invocation_id,
        "run_nonce": facade._run_nonce,
        "attempt_nonce": facade._attempt_nonce,
        "env_step": facade._env_steps,
        "resolved_camera": "head",
        "frame_id": f"authorization:{invocation_id}",
    }
    facade._frame_cache = SimpleNamespace(
        get_current=lambda _camera, frame_id: SimpleNamespace(frame_id=frame_id)
    )

    def switch(target, **_kwargs):
        before = facade._controller_state
        facade._controller_state = target
        return {"from": before, "to": target, "changed": before != target}

    facade._switch_controller = switch
    facade._sanitized_capability_summary = lambda: {
        "attachments": {
            "available": True,
            "count": 1,
            "by_hand": {
                "left": {"attached": held_hand == "left"},
                "right": {"attached": held_hand == "right"},
            },
            "conflict": False,
        },
        "gripper_state": {"left": "unknown", "right": "unknown"},
    }
    return facade, attachment_state


def _review_frame(frame_id: str, *, env_step: int) -> SimpleNamespace:
    return SimpleNamespace(
        frame_id=frame_id,
        capture_group_id=f"capture:{frame_id}",
        step_index=env_step,
    )


def _record_review(
    facade: BehaviorEnvFacade,
    *,
    frame_id: str,
    assessment: str,
) -> dict:
    frame = _review_frame(frame_id, env_step=facade._env_steps)
    facade._public_capture_sequence += 1
    capture_receipt = facade._seal_attempt_receipt(
        {
            "kind": "public_observe_capture",
            "requested_camera": "head",
            "resolved_camera": "head",
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "env_step": facade._env_steps,
            "capture_sequence": facade._public_capture_sequence,
            "rgb_sha256": "0" * 64,
        }
    )
    return facade._record_frame_review_cycle(
        requested_camera="head",
        resolved_camera="head",
        frame=frame,
        capture_receipt=capture_receipt,
        assessment=assessment,
    )


def _start_rotate_pi0_cycle(
    facade: BehaviorEnvFacade,
    *,
    call_index: int,
) -> None:
    held_hand = facade._attachment_runtime_facts()["hands"][0]
    rotate = facade.rotate_wrist(
        hand=held_hand,
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, held_hand),
    )
    rotate_receipt = rotate["attached_rotate_receipt"]
    assert rotate_receipt["resolved_hand"] in {"left", "right"}
    assert rotate_receipt["started_env_step"] <= rotate_receipt["completed_env_step"]
    target_review = _record_review(
        facade,
        frame_id=f"target:{call_index}",
        assessment="target_bearing_surface_confirmed",
    )
    assert target_review["qualifying_pre_vla_target_review"] is True

    visual_check = {
        "camera": "head",
        "frame_id": f"authorization:{call_index}",
        "assessment": "current_task_object_configuration_reviewed",
    }
    preflight = facade.prepare_vla_invocation(
        invocation_id=f"call-{call_index}",
        call_index=call_index,
        vla_status=None,
        current_object_visual_check=visual_check,
    )
    assert preflight["primitive_success"] is True
    confirmed = facade.prepare_vla_invocation(
        invocation_id=f"call-{call_index}",
        call_index=call_index,
        vla_status={"actions_enabled": True},
        current_object_visual_check=visual_check,
    )
    assert confirmed["primitive_success"] is True
    facade._env_steps += 32
    facade._record_full_pi0_chunk_for_rotate_candidate(
        executed_steps=32,
        terminated=False,
        truncated=False,
    )
    finalized = facade.finalize_paused_runtime(
        {
            "actions_enabled": False,
            "healthz": {"actions_enabled": False, "pid": 123},
            "endpoint": "http://127.0.0.1:9999",
        }
    )
    assert finalized["opposite_surface_review_pending"] is True


def _complete_rotate_pi0_cycle(
    facade: BehaviorEnvFacade,
    *,
    call_index: int,
) -> dict:
    _start_rotate_pi0_cycle(facade, call_index=call_index)
    return _record_review(
        facade,
        frame_id=f"opposite:{call_index}",
        assessment="opposite_surface_confirmed",
    )


def test_two_distinct_full_cycles_latch_pi0_for_either_held_arm(tmp_path):
    failure = "pi0_nav_pick_disabled_by_opposite_surface_receipt"
    for held_hand in ("left", "right"):
        facade, _attachment = _opposite_surface_runtime(
            tmp_path / held_hand,
            held_hand=held_hand,
        )

        first = _complete_rotate_pi0_cycle(facade, call_index=1)
        assert first["qualifying_cycle"] is True
        assert first["completed_qualifying_cycles"] == 1
        assert first["pi0_nav_pick_disabled"] is False
        assert facade._public_capture_sequence == 2
        assert (
            failure
            not in facade.guard_tool_call(
                name="pi0_nav_pick",
                input_dict={"current_object_visual_check": {}},
            )["failed_preconditions"]
        )

        second = _complete_rotate_pi0_cycle(facade, call_index=2)
        assert second["qualifying_cycle"] is True
        assert second["completed_qualifying_cycles"] == 2
        assert second["pi0_nav_pick_disabled"] is True
        assert facade._public_capture_sequence == 4
        cycle_ids = [
            item["cycle_id"]
            for item in second["pi0_nav_pick_disable_receipt"]["cycle_receipts"]
        ]
        assert len(set(cycle_ids)) == 2
        assert all(
            item["resolved_hand"] == held_hand
            for item in facade._completed_opposite_surface_cycles
        )

        env_steps = facade._env_steps
        third = facade.prepare_vla_invocation(
            invocation_id="call-3",
            call_index=3,
            vla_status=None,
            current_object_visual_check={},
        )
        assert third["failed_preconditions"] == [failure]
        assert third["total_env_steps"] == env_steps
        assert facade._active_vla_invocation is None
        assert facade._pending_vla_visual_authorization is None

        for name, input_dict in (
            ("rotate_wrist", {"hand": held_hand}),
            ("move_to", {"hand": held_hand}),
            (
                "open",
                {"hand": "right" if held_hand == "left" else "left"},
            ),
        ):
            guarded = facade.guard_tool_call(name=name, input_dict=input_dict)
            assert failure not in guarded["failed_preconditions"]


def test_cross_cycle_held_hand_or_attachment_drift_resets_streak(tmp_path):
    hand_drift, attachment = _opposite_surface_runtime(
        tmp_path / "hand-drift",
        held_hand="left",
    )
    first = _complete_rotate_pi0_cycle(hand_drift, call_index=1)
    assert first["completed_qualifying_cycles"] == 1
    right_root = attachment["root"]

    def right_hand_facts():
        return _attachment_facts(
            right={"right_eef_link": right_root},
        )

    hand_drift._attachment_runtime_facts = right_hand_facts
    second = _complete_rotate_pi0_cycle(hand_drift, call_index=2)
    assert second["completed_qualifying_cycles"] == 1
    assert second["pi0_nav_pick_disabled"] is False

    object_drift, attachment = _opposite_surface_runtime(
        tmp_path / "object-drift",
        held_hand="left",
    )
    first = _complete_rotate_pi0_cycle(object_drift, call_index=1)
    assert first["completed_qualifying_cycles"] == 1
    attachment["root"] = SimpleNamespace(prim_path="/World/different_radio")
    second = _complete_rotate_pi0_cycle(object_drift, call_index=2)
    assert second["completed_qualifying_cycles"] == 1
    assert second["pi0_nav_pick_disabled"] is False


def test_disable_receipt_is_not_latched_when_atomic_write_fails(
    tmp_path,
    monkeypatch,
):
    facade, _attachment = _opposite_surface_runtime(
        tmp_path,
        held_hand="left",
    )
    _complete_rotate_pi0_cycle(facade, call_index=1)
    _start_rotate_pi0_cycle(facade, call_index=2)

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated receipt write failure")

    monkeypatch.setattr(env_server_module, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="simulated receipt write failure"):
        _record_review(
            facade,
            frame_id="opposite:write-failure",
            assessment="opposite_surface_confirmed",
        )

    assert facade._pi0_nav_pick_disable_receipt is None
    assert facade._pi0_nav_pick_is_receipt_disabled() is False


def test_uncertain_review_step_break_and_replay_break_consecutive_chain(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    first = _complete_rotate_pi0_cycle(facade, call_index=1)
    assert first["completed_qualifying_cycles"] == 1

    _start_rotate_pi0_cycle(facade, call_index=2)
    uncertain = _record_review(
        facade,
        frame_id="uncertain:2",
        assessment="side_or_indeterminate",
    )
    assert uncertain["qualifying_cycle"] is False
    assert uncertain["completed_qualifying_cycles"] == 0
    assert uncertain["pi0_nav_pick_disabled"] is False

    _complete_rotate_pi0_cycle(facade, call_index=3)
    facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )
    facade._env_steps += 1
    stepped = _record_review(
        facade,
        frame_id="stepped:4",
        assessment="target_bearing_surface_confirmed",
    )
    assert stepped["qualifying_pre_vla_target_review"] is False
    assert stepped["nonqualifying_reason"] == "target_review_not_at_rotate_env_step"
    assert stepped["completed_qualifying_cycles"] == 0

    facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )
    replay_id = "replayed-before-rotate"
    facade._latest_successful_held_rotate_public_frame_ids.add(replay_id)
    replayed = _record_review(
        facade,
        frame_id=replay_id,
        assessment="target_bearing_surface_confirmed",
    )
    assert replayed["qualifying_pre_vla_target_review"] is False
    assert replayed["nonqualifying_reason"] == (
        "target_review_frame_was_not_captured_after_rotate"
    )


def test_attachment_identity_change_prevents_second_cycle_from_latching(tmp_path):
    facade, attachment = _opposite_surface_runtime(tmp_path, held_hand="right")
    first = _complete_rotate_pi0_cycle(facade, call_index=1)
    assert first["completed_qualifying_cycles"] == 1

    _start_rotate_pi0_cycle(facade, call_index=2)
    attachment["root"] = SimpleNamespace(prim_path="/World/different_object")
    changed = _record_review(
        facade,
        frame_id="opposite:changed",
        assessment="opposite_surface_confirmed",
    )
    assert changed["qualifying_cycle"] is False
    assert changed["nonqualifying_reason"] == "held_attachment_changed_before_review"
    assert changed["completed_qualifying_cycles"] == 0
    assert changed["pi0_nav_pick_disabled"] is False


def test_raw_success_wins_over_opposite_surface_disable_receipt(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    _complete_rotate_pi0_cycle(facade, call_index=1)
    _complete_rotate_pi0_cycle(facade, call_index=2)
    facade._last_info = {"done": {"success": True}}

    guarded = facade.guard_tool_call(
        name="pi0_nav_pick",
        input_dict={"current_object_visual_check": {}},
    )

    assert guarded["task_success"] is True
    assert "official_success_latched" in guarded["failed_preconditions"]
    assert (
        "pi0_nav_pick_disabled_by_opposite_surface_receipt"
        not in guarded["failed_preconditions"]
    )


def test_phase_b_disable_race_clears_invocation_authority(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    preflight = facade.prepare_vla_invocation(
        invocation_id="racing-call",
        call_index=1,
        vla_status=None,
        current_object_visual_check={},
    )
    assert preflight["primitive_success"] is True
    facade._pi0_nav_pick_disable_receipt = facade._seal_attempt_receipt(
        {
            "kind": "pi0_nav_pick_attempt_disable",
            "reason": "two_consecutive_opposite_surface_cycles",
            "env_step": facade._env_steps,
            "cycle_receipts": [
                {"cycle_id": "cycle-one"},
                {"cycle_id": "cycle-two"},
            ],
        }
    )

    rejected = facade.prepare_vla_invocation(
        invocation_id="racing-call",
        call_index=1,
        vla_status={"actions_enabled": True},
        current_object_visual_check={},
    )

    assert rejected["failed_preconditions"] == [
        "pi0_nav_pick_disabled_by_opposite_surface_receipt"
    ]
    assert facade._active_vla_invocation is None
    assert facade._active_vla_call_index is None
    assert facade._pending_vla_visual_authorization is None
    assert facade._active_rotate_pi0_candidate is None


def test_raw_success_phase_b_clears_authority_without_reenabling_vla(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    facade._official_success_receipt = None
    facade._official_success_receipt_path = tmp_path / "official_success_receipt.json"
    facade._finalize_video_segment = lambda: None
    facade._video_error = None
    facade._video_path = tmp_path / "episode.mp4"
    preflight = facade.prepare_vla_invocation(
        invocation_id="success-race",
        call_index=1,
        vla_status=None,
        current_object_visual_check={},
    )
    assert preflight["primitive_success"] is True

    facade._last_info = {"done": {"success": True}}
    confirmed = facade.prepare_vla_invocation(
        invocation_id="success-race",
        call_index=1,
        vla_status={"actions_enabled": True},
        current_object_visual_check={},
    )

    assert confirmed["primitive_success"] is False
    assert confirmed["task_success"] is True
    assert confirmed["stop_reason"] == "official_success_latched"
    assert confirmed["vla_actions_enabled"] is False
    assert facade._active_vla_invocation is None
    assert facade._active_vla_call_index is None
    assert facade._pending_vla_visual_authorization is None
    assert facade._active_rotate_pi0_candidate is None
    assert facade._controller_state == "frozen"


@pytest.mark.parametrize(
    ("controller_mode", "expected_warmup_calls"),
    [("hybrid", 1), ("pi0_nav_pick_only", 0)],
)
def test_reset_clears_opposite_surface_attempt_state(
    tmp_path,
    controller_mode,
    expected_warmup_calls,
):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    facade._controller_mode = controller_mode
    facade._reset_completed = False
    facade._pi0_nav_pick_disable_receipt = {"stale": True}
    facade._completed_opposite_surface_cycles = [{"stale": True}]
    facade._awaiting_opposite_surface_review = {"stale": True}
    facade._active_vla_invocation = "stale-call"
    facade._pending_vla_visual_authorization = {"stale": True}
    facade._env = SimpleNamespace(
        env_reset=lambda: ([{}], [{"done": {"success": False}}]),
        _wrap_obs=lambda _raw: {
            "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
            "states": np.zeros((1, 256), dtype=np.float32),
            "task_descriptions": ["turn on the radio"],
        },
    )
    facade._robot = lambda: SimpleNamespace(_controller_config={"base": {}})
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda _observation: None
    warmup_calls: list[bool] = []
    facade._planner.warmup = lambda: (
        warmup_calls.append(True) or {"status": "complete", "real_plan_only": True}
    )

    facade.reset()

    assert facade._pi0_nav_pick_disable_receipt is None
    assert facade._completed_opposite_surface_cycles == []
    assert facade._awaiting_opposite_surface_review is None
    assert facade._active_vla_invocation is None
    assert facade._pending_vla_visual_authorization is None
    assert len(warmup_calls) == expected_warmup_calls
    if controller_mode == "hybrid":
        assert facade._planner_warmup_report["real_plan_only"] is True
    else:
        assert facade._planner_warmup_report is None


def _public_review_runtime(
    tmp_path,
    *,
    held_hand: str = "left",
) -> tuple[BehaviorEnvFacade, SimpleNamespace, bytes]:
    facade, _attachment = _opposite_surface_runtime(
        tmp_path,
        held_hand=held_hand,
    )
    image_bytes = b"reviewed-rgb"
    frame = SimpleNamespace(
        frame_id="head:public:1",
        capture_group_id="capture:public:1",
        step_index=facade._env_steps,
        timestamp_s=__import__("time").monotonic(),
        intrinsics=SimpleNamespace(
            width=4,
            height=4,
            fx=1.0,
            fy=1.0,
            cx=1.5,
            cy=1.5,
        ),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    facade._frame_cache = SimpleNamespace(
        latest=lambda _camera: frame,
        get_current=lambda _camera, frame_id: (
            frame
            if frame_id == frame.frame_id
            else (_ for _ in ()).throw(RuntimeError("unexpected frame"))
        ),
        ttl_s=60.0,
    )
    facade._planner = SimpleNamespace(
        observe=lambda camera: {
            "camera": camera,
            "frame_id": frame.frame_id,
            "capture_group": {"id": frame.capture_group_id, "cameras": ["head"]},
            "metrics": {},
            "_image_bytes": image_bytes,
        }
    )
    capture = facade.observe(camera="head")
    assert capture["capture_receipt"]["capture_sequence"] == 1
    assert (
        capture["capture_receipt"]["rgb_sha256"]
        == __import__("hashlib").sha256(image_bytes).hexdigest()
    )
    return facade, frame, image_bytes


def _reference_transform(position) -> list[list[float]]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform.tolist()


def _frame_bound_hand_references(
    *,
    left: dict[str, object] | None = None,
    right: dict[str, object] | None = None,
) -> dict[str, object]:
    symmetric = {
        "palm": _reference_transform([0.0, 0.0, -0.90]),
        "grip_point": _reference_transform([0.0, 0.0, -0.95]),
        "finger_roots": [
            _reference_transform([0.0, 0.03, -1.0]),
            _reference_transform([0.0, -0.04, -1.0]),
        ],
    }
    return {
        "schema_version": 1,
        "available": True,
        "env_step": 0,
        "source": "capture_time_live_r1pro_link_transforms",
        "hands": {
            "left": left or symmetric,
            "right": right or symmetric,
        },
    }


def _hand_geometry_sync_certificate(
    *,
    synchronized: bool = True,
    selected_hand_passed: bool = True,
    translation_error_m: float = 0.0,
    rotation_error_deg: float = 0.0,
    finger_joint_error_m: float = 0.0,
) -> dict[str, object]:
    hand_certificate = {
        "passed": selected_hand_passed,
        "camera_pose_source": "sensor_cameraViewTransform",
        "camera_pose_render_bound": True,
        "palm_from_camera": {
            "translation_error_m": translation_error_m,
            "rotation_error_rad": np.deg2rad(rotation_error_deg),
            "rotation_error_deg": rotation_error_deg,
            "passed": selected_hand_passed,
        },
        "grip_point_from_camera": {
            "translation_error_m": translation_error_m,
            "rotation_error_rad": np.deg2rad(rotation_error_deg),
            "rotation_error_deg": rotation_error_deg,
            "passed": selected_hand_passed,
        },
        "finger_joint_capture_match": {
            "max_abs_error_m": finger_joint_error_m,
            "passed": selected_hand_passed,
        },
    }
    return {
        "schema_version": 1,
        "available": True,
        "synchronized": synchronized,
        "reason": None if synchronized else "hand_geometry_sync_residual_failed",
        "env_step": 0,
        "render_sync_iterations": 3,
        "translation_tolerance_m": 0.001,
        "rotation_tolerance_deg": 0.25,
        "finger_joint_tolerance_m": 0.0001,
        "source": "render_sync_plus_official_r1pro_fixed_extrinsics",
        "hands": {
            "left": dict(hand_certificate),
            "right": dict(hand_certificate),
        },
    }


def _test_world_transform(
    translation: tuple[float, float, float],
    *,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = translation
    return transform


def _fake_r1pro_capture_inputs():
    facade = _runtime()
    camera_to_world = {
        "head": _test_world_transform((0.0, 0.0, 1.4)),
        "left_wrist": _test_world_transform((0.3, 0.2, 1.0), yaw_rad=0.35),
        "right_wrist": _test_world_transform((-0.4, 0.1, 0.9), yaw_rad=-0.45),
    }
    fixed = r1pro_wrist_camera_reference_transforms()
    links: dict[str, np.ndarray] = {}
    eef_names: dict[str, str] = {}
    finger_names: dict[str, list[str]] = {}
    for hand in ("left", "right"):
        wrist_camera = camera_to_world[f"{hand}_wrist"]
        palm = wrist_camera @ np.linalg.inv(fixed["palm_from_camera"])
        grip = wrist_camera @ np.linalg.inv(fixed["grip_point_from_camera"])
        eef_name = f"{hand}_eef_link"
        current_finger_names = [
            f"{hand}_gripper_finger_link1",
            f"{hand}_gripper_finger_link2",
        ]
        links[f"{hand}_gripper_link"] = palm
        links[eef_name] = grip
        links[current_finger_names[0]] = palm @ _test_world_transform(
            (0.0, 0.018, -0.038)
        )
        links[current_finger_names[1]] = palm @ _test_world_transform(
            (0.0, -0.018, -0.038)
        )
        eef_names[hand] = eef_name
        finger_names[hand] = current_finger_names

    qpos = np.asarray([0.012, 0.013, 0.021, 0.022], dtype=np.float64)
    robot = SimpleNamespace(
        links=links,
        eef_link_names=eef_names,
        finger_link_names=finger_names,
        gripper_control_idx={"left": [0, 1], "right": [2, 3]},
        get_joint_positions=lambda: qpos.copy(),
    )
    facade._robot = lambda: robot
    facade._live_link_world_transform = lambda link, *, reference: np.asarray(
        link, dtype=np.float64
    ).copy()
    raw_proprio = np.zeros(256, dtype=np.float64)
    raw_proprio[RAW_PROPRIO_SEGMENTS["left_gripper"]] = qpos[:2]
    raw_proprio[RAW_PROPRIO_SEGMENTS["right_gripper"]] = qpos[2:]
    pose_lineage = {
        camera: {
            "source": (
                "payload_view_matrix"
                if camera == "head"
                else "sensor_cameraViewTransform"
            ),
            "render_bound": True,
            "env_step": 0,
            "render_sync_iterations": 3,
        }
        for camera in ("head", "left_wrist", "right_wrist")
    }
    return facade, robot, raw_proprio, camera_to_world, pose_lineage


def _depth_probe_runtime(
    *,
    task_spec=PICKING_UP_TRASH_TASK_SPEC,
    requested_camera: str = "left_wrist",
    references: dict[str, object] | None = None,
    correction: CameraCorrectionProfile | None = None,
    sync_certificate: dict[str, object] | None = None,
    include_sync_certificate: bool = True,
):
    facade = _runtime()
    facade._task_spec = task_spec
    attachment_state: dict[str, object] = {}

    def attachment_facts():
        return _attachment_facts()

    facade._attachment_runtime_facts = attachment_facts
    cache = FrameCache(ttl_s=60.0)
    if correction is not None:
        cache.set_correction_profile(correction.camera, correction)
    intrinsics = CameraIntrinsics(
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=4.0,
        width=9,
        height=9,
    )
    capture_metadata = {
        "r1pro_hand_reference_transforms": (
            references if references is not None else _frame_bound_hand_references()
        ),
        "camera_pose_lineage": {
            camera: {
                "source": (
                    "sensor_cameraViewTransform"
                    if camera.endswith("_wrist")
                    else "payload_view_matrix"
                ),
                "render_bound": True,
                "env_step": 0,
                "render_sync_iterations": 3,
            }
            for camera in ("head", "left_wrist", "right_wrist")
        },
        "render_sync_iterations": 3,
    }
    if include_sync_certificate:
        capture_metadata["hand_geometry_sync_certificate"] = (
            sync_certificate
            if sync_certificate is not None
            else _hand_geometry_sync_certificate()
        )
    cache.add_capture_group(
        frames={
            camera: {
                "rgb": np.zeros((9, 9, 3), dtype=np.uint8),
                "depth_m": np.ones((9, 9), dtype=np.float32),
                "intrinsics": intrinsics,
                "camera_to_world": np.eye(4, dtype=np.float64),
            }
            for camera in ("head", "left_wrist", "right_wrist")
        },
        step_index=0,
        capture_group_id="capture:0:hand-distance",
        capture_metadata=capture_metadata,
    )
    observe_calls: list[str] = []

    def observe(camera: str):
        observe_calls.append(camera)
        return cache.observe_payload(camera)

    facade._frame_cache = cache
    facade._planner = SimpleNamespace(observe=observe)
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
        "gripper_state": {"left": "unknown", "right": "unknown"},
    }
    sync_render_calls: list[str] = []
    sync_record_calls: list[dict[str, object]] = []
    facade._render_only_for_hand_geometry = lambda: sync_render_calls.append("render")
    facade._env = SimpleNamespace(
        omnigibson_env=SimpleNamespace(get_obs=lambda: ({}, {})),
        _wrap_obs=lambda _raw: {
            "main_images": np.zeros((1, 9, 9, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 2, 9, 9, 3), dtype=np.uint8),
            "states": np.zeros((1, 256), dtype=np.float64),
            "task_descriptions": ["test"],
        },
    )

    def record(_raw, _observation, **kwargs):
        sync_record_calls.append(kwargs)

    facade._record_rgbd_frames = record
    facade._test_sync_render_calls = sync_render_calls
    facade._test_sync_record_calls = sync_record_calls
    capture = facade.observe(camera=requested_camera)
    return facade, cache, capture, attachment_state, observe_calls


def _probe_for_capture(capture: dict[str, object]) -> dict[str, object]:
    return {
        "frame_id": capture["frame_id"],
        "u": 4,
        "v": 4,
        "depth_window_px": 3,
        "assessment": "target_point_visually_confirmed",
    }


def test_r1pro_hand_reference_producer_accepts_symmetric_full_se3_capture():
    facade, _robot, raw, camera_to_world, pose_lineage = _fake_r1pro_capture_inputs()

    references, certificate = facade._capture_r1pro_hand_reference_transforms(
        camera_to_world_by_camera=camera_to_world,
        camera_pose_lineage_by_camera=pose_lineage,
        raw_proprio=raw,
        render_sync_iterations=3,
    )

    assert references["available"] is True
    assert certificate["synchronized"] is True
    fixed = r1pro_wrist_camera_reference_transforms()
    for hand in ("left", "right"):
        palm = np.asarray(references["hands"][hand]["palm"], dtype=np.float64)
        grip = np.asarray(references["hands"][hand]["grip_point"], dtype=np.float64)
        wrist_camera = camera_to_world[f"{hand}_wrist"]
        np.testing.assert_allclose(
            np.linalg.inv(palm) @ wrist_camera,
            fixed["palm_from_camera"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.inv(grip) @ wrist_camera,
            fixed["grip_point_from_camera"],
            atol=1e-12,
        )
        selected = certificate["hands"][hand]
        assert selected["passed"] is True
        assert selected["camera_pose_source"] == "sensor_cameraViewTransform"
        assert selected["camera_pose_render_bound"] is True
        assert selected["finger_joint_capture_match"]["max_abs_error_m"] == 0.0


@pytest.mark.parametrize(
    "failure",
    ["untrusted_pose_source", "camera_residual", "finger_q_mismatch"],
)
def test_r1pro_hand_reference_producer_fails_closed_on_capture_mismatch(failure):
    facade, _robot, raw, camera_to_world, pose_lineage = _fake_r1pro_capture_inputs()
    if failure == "untrusted_pose_source":
        pose_lineage["left_wrist"] = {
            **pose_lineage["left_wrist"],
            "source": "live_sensor_pose_fallback",
            "render_bound": False,
        }
    elif failure == "camera_residual":
        camera_to_world["left_wrist"] = camera_to_world["left_wrist"].copy()
        camera_to_world["left_wrist"][0, 3] += 0.002
    else:
        raw[RAW_PROPRIO_SEGMENTS["left_gripper"].start] += 0.001

    references, certificate = facade._capture_r1pro_hand_reference_transforms(
        camera_to_world_by_camera=camera_to_world,
        camera_pose_lineage_by_camera=pose_lineage,
        raw_proprio=raw,
        render_sync_iterations=3,
    )

    assert references["available"] is False
    assert certificate["synchronized"] is False
    assert certificate["hands"]["left"]["passed"] is False


def test_zero_render_view_uses_untrusted_live_pose_fallback_only():
    class Sensor:
        camera_parameters = {"cameraViewTransform": np.zeros(16)}

        @staticmethod
        def get_position_orientation():
            return np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 1.0])

    matrix, source, render_bound = _runtime()._camera_to_world_with_source(
        camera="left_wrist",
        payload={},
        sensor=Sensor(),
    )

    np.testing.assert_allclose(matrix, np.eye(4), atol=1e-12)
    assert source == "live_sensor_pose_fallback"
    assert render_bound is False


def test_record_rgbd_frames_runs_real_hand_reference_producer_into_frame_metadata():
    facade, _robot, raw, camera_to_world, _pose_lineage = _fake_r1pro_capture_inputs()
    facade._frame_cache = FrameCache(ttl_s=60.0)
    facade._sensor_for_camera = lambda _camera: None
    intrinsics = np.asarray(
        [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    sensor_names = {
        "head": "robot_r1:zed_link:Camera:0",
        "left_wrist": "robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    }
    raw_sensors = {
        sensor_names[camera]: {
            "rgb": np.zeros((9, 9, 3), dtype=np.uint8),
            "depth_linear": np.ones((9, 9), dtype=np.float32),
            "intrinsics": intrinsics,
            "view_matrix": np.linalg.inv(transform).T,
        }
        for camera, transform in camera_to_world.items()
    }

    facade._record_rgbd_frames(
        [{"robot_r1": raw_sensors}],
        {"states": raw},
        strict=True,
        synchronize_hand_geometry=True,
        render_sync_iterations=3,
    )

    for camera in ("left_wrist", "right_wrist"):
        metadata = facade._frame_cache.latest(camera).capture_metadata
        assert metadata["r1pro_hand_reference_transforms"]["available"] is True
        assert metadata["hand_geometry_sync_certificate"]["synchronized"] is True
        assert metadata["camera_pose_lineage"][camera] == {
            "source": "payload_view_matrix",
            "render_bound": True,
            "env_step": 0,
            "render_sync_iterations": 3,
        }


@pytest.mark.parametrize(
    "task_spec",
    [TURNING_ON_RADIO_TASK_SPEC, PICKING_UP_TRASH_TASK_SPEC],
)
@pytest.mark.parametrize(
    ("requested_camera", "expected_hand"),
    [
        ("left_wrist", "left"),
        ("right_wrist", "right"),
    ],
)
def test_wrist_depth_probe_returns_frame_bound_live_hand_distances_for_both_tasks(
    task_spec,
    requested_camera,
    expected_hand,
):
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        task_spec=task_spec,
        requested_camera=requested_camera,
    )
    original_radio_review_receipt = facade._latest_unconsumed_public_capture_receipt
    controller_before = facade._controller_state
    env_step_before = facade._env_steps

    result = facade.observe(
        camera=requested_camera,
        depth_probe=_probe_for_capture(capture),
    )
    probe = result["depth_probe"]

    assert probe["target_point_camera_xyz_m"] == pytest.approx([0.0, 0.0, -1.0])
    assert probe["target_to_palm_m"] == pytest.approx(0.10)
    assert probe["target_to_grip_point_m"] == pytest.approx(0.05)
    assert probe["target_to_finger_roots_m"] == pytest.approx(0.03)
    assert "target_point_world_xyz_m" not in probe
    assert "surface_normal" not in probe
    assert "projection_id" not in probe
    assert probe["semantic_target_verified"] is False
    assert probe["motion_authorization"] is False
    hand_geometry = probe["hand_geometry"]
    assert hand_geometry == {
        "available": True,
        "reason": None,
        "geometry_sha256": facade._latest_public_observation_lineage["geometry_sha256"],
        "frame_id": capture["frame_id"],
        "capture_group_id": "capture:0:hand-distance",
        "env_step": 0,
        "source": "frame_bound_live_r1pro_link_transforms",
        "target_point_camera_frame": "effective_usd_camera",
        "camera_axes": "+X right,+Y up,-Z forward",
        "distance_computation_frame": "world",
        "guidance_only": True,
        "semantic_target_verified": False,
        "collision_authorization": False,
        "close_authorization": False,
        "open_authorization": False,
        "resolved_hand": expected_hand,
        "sync_certificate": {
            "synchronized": True,
            "render_sync_iterations": 3,
            "translation_tolerance_m": 0.001,
            "rotation_tolerance_deg": 0.25,
            "finger_joint_tolerance_m": 0.0001,
            "selected_hand_passed": True,
            "camera_pose_source": "sensor_cameraViewTransform",
            "camera_pose_render_bound": True,
        },
        "target_to_finger_roots_individual_m": pytest.approx([0.03, 0.04]),
    }
    assert facade._env_steps == env_step_before
    assert facade._controller_state == controller_before
    assert observe_calls == [
        f"{expected_hand}_wrist",
        f"{expected_hand}_wrist",
    ]
    assert facade._test_sync_render_calls == ["render", "render", "render"]
    assert facade._test_sync_record_calls == [
        {
            "strict": True,
            "synchronize_hand_geometry": True,
            "render_sync_iterations": 3,
        }
    ]
    assert (
        facade._latest_unconsumed_public_capture_receipt
        == original_radio_review_receipt
    )


def test_head_depth_probe_returns_camera_point_but_no_guessed_hand_distances():
    facade, _cache, capture, _attachment, _calls = _depth_probe_runtime(
        requested_camera="head",
    )

    probe = facade.observe(
        camera="head",
        depth_probe=_probe_for_capture(capture),
    )["depth_probe"]

    assert probe["target_point_camera_xyz_m"] == pytest.approx([0.0, 0.0, -1.0])
    for key in (
        "target_to_palm_m",
        "target_to_grip_point_m",
        "target_to_finger_roots_m",
    ):
        assert key not in probe
    assert probe["hand_geometry"]["available"] is False
    assert probe["hand_geometry"]["reason"] == "requires_resolved_wrist_camera"
    assert "resolved_hand" not in probe["hand_geometry"]
    assert "target_to_finger_roots_individual_m" not in probe["hand_geometry"]
    assert facade._test_sync_render_calls == []
    assert facade._test_sync_record_calls == []
    for authorization in (
        "semantic_target_verified",
        "collision_authorization",
        "close_authorization",
        "open_authorization",
    ):
        assert probe["hand_geometry"][authorization] is False


def test_wrist_depth_probe_selects_only_the_requested_physical_hand_geometry():
    left = {
        "palm": _reference_transform([0.0, 0.0, -0.9]),
        "grip_point": _reference_transform([0.0, 0.0, -0.8]),
        "finger_roots": [
            _reference_transform([0.0, 0.03, -1.0]),
            _reference_transform([0.0, -0.04, -1.0]),
        ],
    }
    right = {
        "palm": _reference_transform([0.0, 0.0, -0.6]),
        "grip_point": _reference_transform([0.0, 0.0, -0.5]),
        "finger_roots": [
            _reference_transform([0.0, 0.30, -1.0]),
            _reference_transform([0.0, -0.40, -1.0]),
        ],
    }
    references = _frame_bound_hand_references(left=left, right=right)

    left_facade, _cache, left_capture, _state, _calls = _depth_probe_runtime(
        requested_camera="left_wrist",
        references=references,
    )
    left_probe = left_facade.observe(
        camera="left_wrist",
        depth_probe=_probe_for_capture(left_capture),
    )["depth_probe"]
    right_facade, _cache, right_capture, _state, _calls = _depth_probe_runtime(
        requested_camera="right_wrist",
        references=references,
    )
    right_probe = right_facade.observe(
        camera="right_wrist",
        depth_probe=_probe_for_capture(right_capture),
    )["depth_probe"]

    assert left_probe["hand_geometry"]["resolved_hand"] == "left"
    assert left_probe["target_to_palm_m"] == pytest.approx(0.1)
    assert left_probe["target_to_finger_roots_m"] == pytest.approx(0.03)
    assert right_probe["hand_geometry"]["resolved_hand"] == "right"
    assert right_probe["target_to_palm_m"] == pytest.approx(0.4)
    assert right_probe["target_to_finger_roots_m"] == pytest.approx(0.3)


def test_wrist_depth_probe_uses_the_capture_snapshot_after_live_links_move():
    references = _frame_bound_hand_references()
    facade, cache, capture, _attachment, _calls = _depth_probe_runtime(
        references=references,
    )
    # Mutating the source object models later live articulation. FrameCache
    # deep-copies capture metadata, so the frame-bound snapshot must not move.
    references["hands"]["left"]["finger_roots"][0][1][3] = 0.9
    facade._robot = lambda: (_ for _ in ()).throw(
        AssertionError("probe must not re-read live robot links")
    )

    probe = facade.observe(
        camera="left_wrist",
        depth_probe=_probe_for_capture(capture),
    )["depth_probe"]

    assert probe["target_to_finger_roots_m"] == pytest.approx(0.03)
    assert cache.latest("left_wrist").capture_metadata[
        "r1pro_hand_reference_transforms"
    ]["hands"]["left"]["finger_roots"][0][1][3] == pytest.approx(0.03)


def test_wrist_depth_probe_reports_the_effective_corrected_camera_point():
    correction_matrix = np.eye(4, dtype=np.float64)
    correction_matrix[:3, 3] = [0.04, -0.02, 0.01]
    correction = CameraCorrectionProfile(
        camera="left_wrist",
        raw_to_corrected_camera=correction_matrix,
        enabled=True,
        metrics={"enabled": True, "reason": "test"},
    )
    facade, _cache, capture, _attachment, _calls = _depth_probe_runtime(
        requested_camera="left_wrist",
        correction=correction,
    )

    probe = facade.observe(
        camera="left_wrist",
        depth_probe=_probe_for_capture(capture),
    )["depth_probe"]

    assert probe["target_point_camera_xyz_m"] == pytest.approx([0.04, -0.02, -0.99])
    assert probe["hand_geometry"]["target_point_camera_frame"] == (
        "effective_usd_camera"
    )
    assert probe["hand_geometry"]["distance_computation_frame"] == "world"


@pytest.mark.parametrize(
    "tamper",
    ["intrinsics", "camera_to_world", "correction", "hand_references"],
)
def test_depth_probe_rejects_any_frame_geometry_digest_change(tamper):
    facade, cache, capture, _attachment, observe_calls = _depth_probe_runtime()
    frame = cache.latest("left_wrist")
    if tamper == "intrinsics":
        frame.intrinsics = CameraIntrinsics(
            fx=4.1,
            fy=4.0,
            cx=4.0,
            cy=4.0,
            width=9,
            height=9,
        )
    elif tamper == "camera_to_world":
        frame.camera_to_world[0, 3] += 0.01
    elif tamper == "correction":
        changed = np.eye(4, dtype=np.float64)
        changed[1, 3] = 0.01
        frame.correction_profile = CameraCorrectionProfile(
            camera="left_wrist",
            raw_to_corrected_camera=changed,
            enabled=True,
            metrics={"enabled": True, "reason": "tampered"},
        )
    else:
        frame.capture_metadata["r1pro_hand_reference_transforms"]["hands"]["left"][
            "palm"
        ][0][3] += 0.01

    with pytest.raises(RuntimeError, match="frame-geometry lineage changed"):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )

    assert facade._env_steps == 0
    assert observe_calls == ["left_wrist"]


def test_wrist_depth_probe_rejects_missing_or_nonfinite_capture_hand_geometry():
    missing = {
        "schema_version": 1,
        "available": False,
        "reason": "r1pro_live_hand_geometry_unavailable",
        "env_step": 0,
        "source": "capture_time_live_r1pro_link_transforms",
        "hands": {},
    }
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        references=missing,
    )
    with pytest.raises(RuntimeError, match="hand-reference geometry is unavailable"):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )
    assert observe_calls == ["left_wrist"]

    nonfinite = _frame_bound_hand_references()
    nonfinite["hands"]["left"]["palm"][0][3] = float("nan")
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        references=nonfinite,
    )
    with pytest.raises(CameraGeometryError, match="NaN or infinity"):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )
    assert observe_calls == ["left_wrist"]


@pytest.mark.parametrize(
    ("certificate", "include_certificate"),
    [
        (None, False),
        (
            _hand_geometry_sync_certificate(synchronized=False),
            True,
        ),
        (
            _hand_geometry_sync_certificate(
                translation_error_m=0.001001,
            ),
            True,
        ),
        (
            _hand_geometry_sync_certificate(
                rotation_error_deg=0.250001,
            ),
            True,
        ),
        (
            _hand_geometry_sync_certificate(
                finger_joint_error_m=0.0001001,
            ),
            True,
        ),
    ],
)
def test_wrist_depth_probe_rejects_missing_false_or_out_of_tolerance_sync_certificate(
    certificate,
    include_certificate,
):
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        sync_certificate=certificate,
        include_sync_certificate=include_certificate,
    )

    with pytest.raises(
        RuntimeError,
        match="hand-geometry sync certificate is unavailable",
    ):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )

    assert facade._env_steps == 0
    assert facade._controller_state == _CONTROLLER_VLA
    assert observe_calls == ["left_wrist"]


@pytest.mark.parametrize(
    ("component", "mutation"),
    [
        ("palm_from_camera", "missing_rotation_rad"),
        ("palm_from_camera", "nan_rotation_rad"),
        ("palm_from_camera", "inconsistent_rotation_units"),
        ("grip_point_from_camera", "missing_rotation_rad"),
        ("grip_point_from_camera", "nan_rotation_rad"),
        ("grip_point_from_camera", "inconsistent_rotation_units"),
    ],
)
def test_wrist_depth_probe_rejects_unverifiable_rotation_residuals(
    component,
    mutation,
):
    certificate = copy.deepcopy(_hand_geometry_sync_certificate())
    residual = certificate["hands"]["left"][component]
    if mutation == "missing_rotation_rad":
        del residual["rotation_error_rad"]
    elif mutation == "nan_rotation_rad":
        residual["rotation_error_rad"] = float("nan")
    else:
        residual["rotation_error_rad"] = 0.0
        residual["rotation_error_deg"] = 0.1
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        sync_certificate=certificate,
    )

    with pytest.raises(
        RuntimeError,
        match="hand-geometry sync certificate is unavailable",
    ):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )

    assert observe_calls == ["left_wrist"]


@pytest.mark.parametrize(
    ("pose_source", "render_bound"),
    [
        ("live_sensor_pose_fallback", False),
        ("sensor_cameraViewTransform", False),
        ("payload_pose_matrix_fallback", True),
    ],
)
def test_wrist_depth_probe_rejects_non_render_bound_camera_pose_certificate(
    pose_source,
    render_bound,
):
    certificate = copy.deepcopy(_hand_geometry_sync_certificate())
    certificate["hands"]["left"]["camera_pose_source"] = pose_source
    certificate["hands"]["left"]["camera_pose_render_bound"] = render_bound
    facade, _cache, capture, _attachment, observe_calls = _depth_probe_runtime(
        sync_certificate=certificate,
    )

    with pytest.raises(
        RuntimeError,
        match="hand-geometry sync certificate is unavailable",
    ):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )

    assert observe_calls == ["left_wrist"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("attempt", "immediately preceding"),
        ("env_step", "immediately preceding"),
        ("resolved_camera", "immediately preceding"),
    ],
)
def test_wrist_depth_probe_rejects_wrong_attempt_step_or_dynamic_camera_lineage(
    mutation,
    error,
):
    facade, _cache, capture, attachment, observe_calls = _depth_probe_runtime(
        requested_camera="left_wrist",
    )
    if mutation == "attempt":
        facade._attempt_nonce = "different-attempt"
    elif mutation == "env_step":
        facade._env_steps += 1
    else:
        facade._latest_public_observation_lineage["resolved_camera"] = "right_wrist"

    with pytest.raises(RuntimeError, match=error):
        facade.observe(
            camera="left_wrist",
            depth_probe=_probe_for_capture(capture),
        )

    assert observe_calls == ["left_wrist"]


def test_frame_review_consumes_only_immediately_preceding_same_camera_capture(
    tmp_path,
):
    mismatch, frame, _image = _public_review_runtime(tmp_path / "mismatch")
    try:
        mismatch.observe(
            camera="left_wrist",
            frame_review={
                "frame_id": frame.frame_id,
                "assessment": "side_or_indeterminate",
            },
        )
    except RuntimeError as exc:
        assert "immediately preceding, same-camera" in str(exc)
    else:
        raise AssertionError("cross-camera frame review was accepted")

    facade, frame, _image = _public_review_runtime(tmp_path / "replay")
    accepted = facade.observe(
        camera="head",
        frame_review={
            "frame_id": frame.frame_id,
            "assessment": "side_or_indeterminate",
        },
    )
    assert accepted["frame_review"]["accepted"] is True
    assert facade._latest_unconsumed_public_capture_receipt is None
    try:
        facade.observe(
            camera="head",
            frame_review={
                "frame_id": frame.frame_id,
                "assessment": "side_or_indeterminate",
            },
        )
    except RuntimeError as exc:
        assert "immediately preceding, same-camera" in str(exc)
    else:
        raise AssertionError("consumed frame review authority was replayed")


def test_capture_or_review_after_target_review_invalidates_pre_vla_authority(
    tmp_path,
):
    for interruption in ("capture", "review"):
        facade, _attachment = _opposite_surface_runtime(
            tmp_path / interruption,
            held_hand="left",
        )
        facade.rotate_wrist(
            hand="left",
            relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
            visual_hand_check=_fresh_visual_hand_check(facade, "left"),
        )
        target = _record_review(
            facade,
            frame_id=f"target:{interruption}",
            assessment="target_bearing_surface_confirmed",
        )
        assert target["qualifying_pre_vla_target_review"] is True

        if interruption == "capture":
            frame = _review_frame("extra:capture", env_step=facade._env_steps)
            facade._register_public_capture(
                requested_camera="head",
                resolved_camera="head",
                frame=frame,
                image_bytes=b"extra",
            )
        else:
            try:
                facade._review_public_observation(
                    requested_camera="head",
                    frame_review={
                        "frame_id": "extra:review",
                        "assessment": "side_or_indeterminate",
                    },
                )
            except RuntimeError as exc:
                assert "invalidated the pending target-surface" in str(exc)
            else:
                raise AssertionError("extra frame review preserved target authority")

        assert facade._held_rotate_target_surface_review is None
        assert facade._latest_successful_held_rotate_receipt is None
        assert facade._completed_opposite_surface_cycles == []


def test_scene_change_invalidates_awaiting_review_and_between_cycle_streak(
    tmp_path,
):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="right")
    facade._planner._gripper_command = lambda *_args, **_kwargs: {
        "primitive_success": True,
        "task_success": False,
        "stop_reason": "opened",
        "metrics": _successful_gripper_isolation("left"),
    }

    _start_rotate_pi0_cycle(facade, call_index=1)
    assert facade._awaiting_opposite_surface_review is not None
    facade.open(
        hand="left",
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )
    assert facade._awaiting_opposite_surface_review is None
    assert facade._completed_opposite_surface_cycles == []

    first = _complete_rotate_pi0_cycle(facade, call_index=2)
    assert first["completed_qualifying_cycles"] == 1
    facade.open(
        hand="left",
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )
    assert facade._completed_opposite_surface_cycles == []

    after_break = _complete_rotate_pi0_cycle(facade, call_index=3)
    assert after_break["completed_qualifying_cycles"] == 1
    assert after_break["pi0_nav_pick_disabled"] is False


def test_unattached_physical_hand_rotate_never_creates_attached_cycle_receipt(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    first = _complete_rotate_pi0_cycle(facade, call_index=1)
    assert first["completed_qualifying_cycles"] == 1
    press = facade.rotate_wrist(
        hand="right",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "right"),
    )
    assert "attached_rotate_receipt" not in press
    assert facade._latest_successful_held_rotate_receipt is None
    assert facade._completed_opposite_surface_cycles == []

    literal, _calls = _rotate_runtime(held_hand=None)
    result = literal.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_visual_hand_check("left"),
    )
    assert "attached_rotate_receipt" not in result
    assert literal._latest_successful_held_rotate_receipt is None


def test_rotate_on_attached_physical_hand_creates_radio_cycle_receipt(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")

    result = facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )

    assert result["requested_hand"] == "left"
    assert result["resolved_hand"] == "left"
    assert result["attached_rotate_receipt"]["requested_hand"] == "left"
    assert "semantic_role" not in result["attached_rotate_receipt"]
    assert facade._latest_successful_held_rotate_receipt is not None


def test_radio_rotate_receipt_allows_other_hand_to_hold_a_distinct_object(tmp_path):
    facade, attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    other_root = SimpleNamespace(prim_path="/World/open_bin")

    def dual_facts():
        return _attachment_facts(
            left={"left_eef_link": attachment["root"]},
            right={"right_eef_link": other_root},
        )

    facade._attachment_runtime_facts = dual_facts
    result = facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )

    assert result["primitive_success"] is True
    assert result["attached_rotate_receipt"]["resolved_hand"] == "left"
    assert (
        result["metrics"]["attachment_isolation"]["checks"][
            "inactive_attachment_unchanged"
        ]
        is True
    )


def test_later_pi0_and_delayed_next_rotate_break_consecutive_cycles(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="left")
    first = _complete_rotate_pi0_cycle(facade, call_index=1)
    assert first["completed_qualifying_cycles"] == 1

    unbound = facade.prepare_vla_invocation(
        invocation_id="unbound-call",
        call_index=2,
        vla_status=None,
        current_object_visual_check={},
    )
    assert unbound["primitive_success"] is True
    assert facade._active_rotate_pi0_candidate is None
    assert facade._completed_opposite_surface_cycles == []
    facade._clear_active_vla_invocation_state()

    again = _complete_rotate_pi0_cycle(facade, call_index=3)
    assert again["completed_qualifying_cycles"] == 1
    facade._env_steps += 1
    facade.rotate_wrist(
        hand="left",
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_fresh_visual_hand_check(facade, "left"),
    )
    assert facade._completed_opposite_surface_cycles == []


def test_new_pi0_invalidates_pending_opposite_surface_review(tmp_path):
    facade, _attachment = _opposite_surface_runtime(tmp_path, held_hand="right")
    _start_rotate_pi0_cycle(facade, call_index=1)
    assert facade._awaiting_opposite_surface_review is not None

    replacement = facade.prepare_vla_invocation(
        invocation_id="replacement-call",
        call_index=2,
        vla_status=None,
        current_object_visual_check={},
    )

    assert replacement["primitive_success"] is True
    assert facade._awaiting_opposite_surface_review is None
    assert facade._completed_opposite_surface_cycles == []
    assert facade._active_rotate_pi0_candidate is None


def _action_trace_bytes(*records) -> bytes:
    return b"".join(
        (
            record
            if isinstance(record, bytes)
            else json.dumps(record, sort_keys=True).encode("utf-8")
        )
        + b"\n"
        for record in records
    )


def test_action_trace_success_summary_accepts_one_exact_current_success():
    trace = _action_trace_bytes(
        {"step": 7, "info_done": {"success": True}},
    )

    binding = summarize_action_trace_success(trace)

    assert binding == {
        "source": "behavior_action_trace",
        "field_path": "info_done.success",
        "first_success_step": 7,
        "success_interval": [7, 7],
        "success_count": 1,
        "success_later_reverted": False,
        "last_success_step": 7,
        "last_trace_step": 7,
        "final_trace_success": True,
        "action_trace_sha256": hashlib.sha256(trace).hexdigest(),
        "receipt_sha256": None,
        "notes": [],
    }


def test_action_trace_success_summary_latches_and_keeps_first_true_interval():
    trace = _action_trace_bytes(
        {"step": 1, "info_done": {"success": False}},
        {"step": 2, "info_done": {"success": True}},
        {"step": 3, "info_done": {"success": True}},
        {"step": 4, "info_done": {"success": False}},
        {"step": 5, "info_done": {"success": True}},
    )

    binding = summarize_action_trace_success(trace)

    assert binding is not None
    assert binding["first_success_step"] == 2
    assert binding["success_interval"] == [2, 3]
    assert binding["success_count"] == 3
    assert binding["success_later_reverted"] is True
    assert binding["last_success_step"] == 5
    assert binding["last_trace_step"] == 5
    assert binding["final_trace_success"] is True


def test_action_trace_summary_ignores_metadata_and_reports_final_action_step():
    trace = _action_trace_bytes(
        {"event": "init"},
        {"event": "reset"},
        {"event": "step", "step": 4, "info_done": {"success": True}},
        {"event": "step", "step": 5},
    )

    binding = summarize_action_trace_success(trace)

    assert binding is not None
    assert binding["last_trace_step"] == 5
    assert binding["final_trace_success"] is None
    assert all(
        not note.startswith("records_missing_step=") for note in binding["notes"]
    )


def test_nested_legacy_true_never_overrides_current_false():
    trace = _action_trace_bytes(
        {
            "step": 1,
            "info_done": {"success": False},
            "info": {"done": {"success": True}},
        }
    )

    assert summarize_action_trace_success(trace) is None


def test_trace_anomalies_are_not_success_correction_blockers():
    trace = _action_trace_bytes(
        {
            "step": 3,
            "terminated": True,
            "info_done": {"success": False},
        },
        b"{malformed-json",
        {"info_done": {"success": True}},
        {"step": 3, "info_done": {"success": True}},
        {"step": 2, "info_done": {"success": False}},
    )

    binding = summarize_action_trace_success(trace)

    assert binding is not None
    assert binding["first_success_step"] is None
    assert binding["success_interval"] == [None, 3]
    assert binding["success_count"] == 2
    assert binding["success_later_reverted"] is True
    assert binding["last_success_step"] == 3
    assert binding["last_trace_step"] == 2
    assert binding["final_trace_success"] is False
    assert binding["notes"] == [
        "malformed_json_lines=1",
        "records_missing_step=1",
        "duplicate_step_values=1",
        "non_monotonic_step_records=1",
        "success_record_missing_step",
        "termination_observed_before_first_success",
    ]


def test_non_boolean_current_success_does_not_enable_legacy_fallback():
    trace = _action_trace_bytes(
        {
            "step": 9,
            "info_done": {"success": 1},
            "info": {"done": {"success": True}},
        }
    )

    assert summarize_action_trace_success(trace) is None
