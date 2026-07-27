from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.policy_checkpoint import (
    SHARED_POLICY_CHECKPOINT_PATH,
    SHARED_POLICY_PROFILE_ID,
)
from robots.behavior.prompt_bundle import _surface_review_guidance
from robots.behavior.prompts.tasks.picking_up_trash import (
    PICKING_UP_TRASH_PROMPT_NODES,
)
from robots.behavior.prompts.tasks.turning_on_radio import (
    TURNING_ON_RADIO_PROMPT_NODES,
)
from robots.behavior.publication import resolve_publication_identity
from robots.behavior.schemas import BEHAVIOR_TOOL_NAMES, behavior_tool_specs_for_task
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
    get_task_spec,
    resolve_task_spec,
)
from robots.behavior.toolkit import BehaviorToolkit
from rpent.context.prompt_utils import format_prompt


def _attached_rotate_facade(
    *,
    task_spec,
    selected_hand: str,
    other_attached: bool,
) -> BehaviorEnvFacade:
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    other = "right" if selected_hand == "left" else "left"
    roots = {
        selected_hand: {
            f"{selected_hand}_eef_link": SimpleNamespace(
                prim_path=f"/World/{selected_hand}-object"
            )
        }
    }
    if other_attached:
        roots[other] = {
            f"{other}_eef_link": SimpleNamespace(prim_path=f"/World/{other}-object")
        }

    def facts():
        hands = [hand for hand in ("left", "right") if hand in roots]
        return {
            "available": True,
            "attachment_count": len(hands),
            "hands": hands,
            "identity_conflict": False,
            "attached_objects": dict(roots),
            "by_hand": {
                hand: {"attached": hand in roots} for hand in ("left", "right")
            },
        }

    facade._task_spec = task_spec
    facade._run_nonce = "run"
    facade._attempt_nonce = "attempt"
    facade._attempt_index = 1
    facade._env_steps = 4
    facade._official_success_latched = False
    facade._last_info = {"done": {"success": False}}
    facade._public_observed_frame_ids = {"head:4:selected"}
    facade._latest_public_head_frame_id = "head:4:selected"
    facade._frame_cache = SimpleNamespace(
        get_current=lambda camera, frame_id: SimpleNamespace(
            frame_id=frame_id,
            step_index=4,
            capture_group_id="capture:4:selected",
        )
    )
    facade._attachment_runtime_facts = facts
    facade._switch_controller = lambda _target: {"changed": False}
    facade._invalidate_pi0_visual_regression_chain = lambda **_kwargs: None
    facade._completed_opposite_surface_cycles = []
    facade._latest_successful_held_rotate_receipt = None
    facade._held_rotate_target_surface_review = None
    facade._active_rotate_pi0_candidate = None
    facade._awaiting_opposite_surface_review = None
    facade._planner = SimpleNamespace(
        rotate_wrist=lambda **_kwargs: {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "reached",
            "metrics": {
                "single_arm_isolation": {
                    "available": True,
                    "ok": True,
                    "selected_hand": selected_hand,
                }
            },
        }
    )
    facade._require_planner = lambda: facade._planner
    facade._analytic_public_result = lambda result, **_kwargs: dict(result)
    return facade


def _visual_hand_check(hand: str) -> dict[str, str]:
    return {
        "camera": "head",
        "frame_id": "head:4:selected",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def test_public_and_candidate_instance_identity_is_task_scoped() -> None:
    radio = TURNING_ON_RADIO_TASK_SPEC
    trash = PICKING_UP_TRASH_TASK_SPEC

    assert radio.instance_for_public_seed(0, phase="explore") == 242
    assert trash.instance_for_public_seed(0, phase="explore") == 196

    radio_242 = radio.classify_instance(242)
    trash_242 = trash.classify_instance(242)
    assert (radio_242.kind, radio_242.public_seed) == ("explore", 0)
    assert (trash_242.kind, trash_242.public_seed) == ("candidate", None)

    trash_196 = trash.classify_instance(196)
    radio_196 = radio.classify_instance(196)
    assert (trash_196.kind, trash_196.public_seed) == ("explore", 0)
    assert (radio_196.kind, radio_196.public_seed) == ("candidate", None)


def test_task_name_index_and_public_tags_cannot_cross_tasks() -> None:
    radio = resolve_task_spec(task_name="turning_on_radio", task_index=0)
    trash = resolve_task_spec(task_name="picking_up_trash", task_index=1)

    assert radio.tag(0) == "turning_on_radio_s0"
    assert trash.tag(0) == "picking_up_trash_s0"
    assert get_task_spec("turning_on_radio") is radio
    assert get_task_spec("picking_up_trash") is trash
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_task_spec(task_name="turning_on_radio", task_index=1)
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_task_spec(task_name="picking_up_trash", task_index=0)


def test_task_specs_do_not_embed_pi0_execution_policy() -> None:
    for spec in (TURNING_ON_RADIO_TASK_SPEC, PICKING_UP_TRASH_TASK_SPEC):
        assert not hasattr(spec, "dual_attachment_pi0_chunks")
        assert "dual_attachment_pi0_chunks" not in type(spec).__dataclass_fields__


def test_all_supported_tasks_share_the_general_pi05_checkpoint_profile() -> None:
    assert {
        spec.task_name
        for spec in (
            TURNING_ON_RADIO_TASK_SPEC,
            PICKING_UP_TRASH_TASK_SPEC,
        )
    } == {"turning_on_radio", "picking_up_trash"}
    assert SHARED_POLICY_PROFILE_ID == "pi05-b1kpt50-cs32"
    assert str(SHARED_POLICY_CHECKPOINT_PATH) == (
        "/home/ubuntu/lwb/Models/openpi_comet_pytorch/pi05-b1kpt50-cs32"
    )


def test_both_tasks_share_the_same_task_neutral_navigation_contract() -> None:
    radio_tools = behavior_tool_specs_for_task(TURNING_ON_RADIO_TASK_SPEC)
    trash_tools = behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)

    assert tuple(radio_tools) == tuple(trash_tools) == BEHAVIOR_TOOL_NAMES
    assert BEHAVIOR_TOOL_NAMES[-1] == "navigate_to"
    assert radio_tools["navigate_to"] == trash_tools["navigate_to"]
    serialized = json.dumps(radio_tools["navigate_to"], sort_keys=True)
    for task_specific in (
        "radio",
        "button",
        "tipped",
        "trash",
        "soda",
        "ashcan",
    ):
        assert task_specific not in serialized.lower()
    assert '"hand"' not in serialized
    assert '"role"' not in serialized


def test_selected_attachment_rotation_cycle_remains_radio_task_local() -> None:
    radio_guidance = " ".join(
        _surface_review_guidance(TURNING_ON_RADIO_TASK_SPEC).split()
    )
    trash_guidance = " ".join(
        _surface_review_guidance(PICKING_UP_TRASH_TASK_SPEC).split()
    )

    assert 'literal `hand="left"` or `hand="right"`' in radio_guidance
    assert "matching requested and resolved physical hands" in radio_guidance
    assert "stable call-start attachment-fingerprint receipt" in radio_guidance
    assert "other hand" in radio_guidance
    assert "surface-recovery cycle" in radio_guidance
    assert "effective semantic role `held`" not in radio_guidance
    assert "uniquely resolved physical hand" not in radio_guidance

    for radio_only in (
        "attachment-fingerprint receipt",
        "surface-recovery cycle",
        "target_bearing_surface_confirmed",
        "opposite_surface_confirmed",
    ):
        assert radio_only not in trash_guidance
    assert (
        "No task-specific target/opposite-surface regression policy" in trash_guidance
    )


def test_receptacle_grounding_and_navigation_guidance_is_trash_task_local() -> None:
    trash_guidance = " ".join(format_prompt(PICKING_UP_TRASH_PROMPT_NODES).split())
    radio_guidance = " ".join(format_prompt(TURNING_ON_RADIO_PROMPT_NODES).split())

    for marker in (
        "Operator review of s2 established a general failure mode",
        "exactly one currently visible search or navigation subgoal",
        "raw official success has not been latched",
        'fresh `observe(camera="head")`',
        "before the next scene-changing action",
        "stop blind Pi0 search or navigation",
        "project that current point with `pixel_to_world`",
        "Pass the fresh receipt to `navigate_to`",
        "one bounded pure-base stage",
        'obtain another fresh `observe(camera="head")`',
        "Select whichever literal anatomical hand",
        "no fixed route, waypoint, room or landmark order",
    ):
        assert marker in trash_guidance
        assert marker not in radio_guidance

    for forbidden in (
        'hand="left"',
        'hand="right"',
        "max_chunks",
        "max_vla_chunks_per_call",
        "max_total_vla_chunks",
        "call_chunk_limit",
    ):
        assert forbidden not in trash_guidance


@pytest.mark.parametrize("selected_hand", ["left", "right"])
@pytest.mark.parametrize("other_attached", [False, True])
def test_radio_rotate_receipt_tracks_selected_attachment_with_single_or_dual_holds(
    selected_hand: str,
    other_attached: bool,
) -> None:
    facade = _attached_rotate_facade(
        task_spec=TURNING_ON_RADIO_TASK_SPEC,
        selected_hand=selected_hand,
        other_attached=other_attached,
    )

    result = facade.rotate_wrist(
        hand=selected_hand,
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_visual_hand_check(selected_hand),
    )

    receipt = result["attached_rotate_receipt"]
    assert receipt["requested_hand"] == selected_hand
    assert receipt["resolved_hand"] == selected_hand
    assert len(receipt["attachment_fingerprint"]) == 64
    assert "semantic_role" not in receipt
    assert "held" not in receipt
    assert "/World/" not in repr(receipt)


@pytest.mark.parametrize("selected_hand", ["left", "right"])
@pytest.mark.parametrize("other_attached", [False, True])
def test_trash_rotate_never_creates_radio_attachment_receipt(
    selected_hand: str,
    other_attached: bool,
) -> None:
    facade = _attached_rotate_facade(
        task_spec=PICKING_UP_TRASH_TASK_SPEC,
        selected_hand=selected_hand,
        other_attached=other_attached,
    )

    result = facade.rotate_wrist(
        hand=selected_hand,
        relative_axis_angle=[0.0, 1.0, 0.0, 0.2],
        visual_hand_check=_visual_hand_check(selected_hand),
    )

    assert "attached_rotate_receipt" not in result
    assert facade._latest_successful_held_rotate_receipt is None


def test_dashboard_and_serial_sources_do_not_reintroduce_dynamic_hand_roles() -> None:
    root = Path(__file__).parents[1]
    sources = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "robots/behavior/campaign_explore.py",
            "robots/behavior/candidate_explore.py",
            "robots/behavior/serial_explore.py",
            "robots/behavior/dashboard_sink.py",
            "robots/behavior/dashboard_state.py",
            "rpent/dashboard/index.html",
            "rpent/dashboard/index.zh-cn.html",
        )
    )

    for forbidden in (
        "held_wrist",
        "press_wrist",
        "dynamic_role_availability",
        "semantic_role",
    ):
        assert forbidden not in sources


def test_publication_identity_is_task_scoped_not_instance_scoped() -> None:
    radio = resolve_publication_identity(
        task_name="turning_on_radio",
        task_index=0,
        public_seed=0,
    )
    trash = resolve_publication_identity(
        task_name="picking_up_trash",
        task_index=1,
        public_seed=0,
    )

    assert (radio.native_instance, radio.tag) == (242, "turning_on_radio_s0")
    assert (trash.native_instance, trash.tag) == (196, "picking_up_trash_s0")
    assert radio.core_payload_paths == (
        "recipe_turning_on_radio_s0.jsonl",
        "memory/turning_on_radio.md",
        "memory/turning_on_radio_provenance.json",
    )
    assert trash.core_payload_paths == (
        "recipe_picking_up_trash_s0.jsonl",
        "memory/picking_up_trash.md",
        "memory/picking_up_trash_provenance.json",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_publication_identity(
            task_name="picking_up_trash",
            task_index=0,
            public_seed=0,
        )


def test_unsuccessful_trash_explore_does_not_publish_success_artifacts(
    tmp_path: Path,
) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "output_dir": tmp_path,
            "task_name": "picking_up_trash",
            "public_seed": 0,
            "behavior_phase": "explore",
        }
    )

    assert toolkit.write_recipe("picking_up_trash_s0") is None

    result = json.loads((tmp_path / "behavior_result.json").read_text(encoding="utf-8"))
    assert result["task_success"] is False
    assert result["publication_eligible"] is False
    assert result["recipe_path"] is None
    assert not (tmp_path / "recipe_picking_up_trash_s0.jsonl").exists()
    assert not (tmp_path / "memory" / "picking_up_trash.md").exists()
    assert not (tmp_path / "memory" / "picking_up_trash_provenance.json").exists()
    assert not (tmp_path / "memory" / "turning_on_radio.md").exists()
    failure_root = tmp_path / "attempts" / "picking_up_trash_s0"
    assert not (failure_root / "attempt_001_failed.json").exists()
    assert not (failure_root / "attempt_001_failed.jsonl").exists()


def test_sealed_trash_publication_uses_only_trash_namespace(
    tmp_path: Path,
) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "output_dir": tmp_path,
            "task_name": "picking_up_trash",
            "public_seed": 0,
            "behavior_phase": "explore",
        }
    )
    recipe_path = tmp_path / "recipe_picking_up_trash_s0.jsonl"
    toolkit._write_json_atomic(
        recipe_path,
        toolkit._symbolic_recipe(),
        json_lines=True,
    )
    receipt_file_sha256 = "a" * 64
    source_hashes = {
        "official_success_receipt": receipt_file_sha256,
        "behavior_action_trace": "b" * 64,
        "behavior_tool_trace": "c" * 64,
        "final_result": "d" * 64,
        "run_manifest": "e" * 64,
        "session_manifest": "f" * 64,
    }
    toolkit._publish_task_memory(
        recipe_tag="picking_up_trash_s0",
        recipe_path=recipe_path,
        official_success_receipt={
            "source": 'info["done"]["success"]',
            "run_nonce": toolkit._primitives.run_nonce,
            "attempt_nonce": toolkit._primitives.attempt_nonce,
            "attempt_index": toolkit._primitives.attempt_index,
            "env_step": 32,
            "receipt_sha256": "9" * 64,
            "file_sha256": receipt_file_sha256,
        },
        source_artifacts_sha256=source_hashes,
    )

    memory_path = tmp_path / "memory" / "picking_up_trash.md"
    provenance_path = tmp_path / "memory" / "picking_up_trash_provenance.json"
    assert memory_path.is_file()
    assert provenance_path.is_file()
    assert not (tmp_path / "memory" / "turning_on_radio.md").exists()
    assert "radio" not in memory_path.read_text(encoding="utf-8").lower()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["task"] == "picking_up_trash"
    assert provenance["task_index"] == 1
    assert provenance["activity_definition_id"] == 0
    assert provenance["activity_instance_id"] == 196
    assert provenance["source_tag"] == "picking_up_trash_s0"
