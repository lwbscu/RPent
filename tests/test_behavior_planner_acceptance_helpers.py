from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.planner_executor import _axis_angle_to_quat_xyzw
from robots.behavior.schemas import PLANNER_TOOL_SPECS


def _acceptance_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_planner_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "behavior_planner_acceptance_tested",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _radio_acceptance_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_radio_tool_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "behavior_radio_tool_acceptance_tested",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_marker_matches_rendered_mesh_prim_path():
    module = _acceptance_module()
    segmentation = np.zeros((15, 15), dtype=np.int64)
    segmentation[3:12, 3:12] = 7
    labels = {7: "/World/scene_0/rpent_private_camera_marker/base_link/visuals"}

    assert module._marker_interior_pixel(segmentation, labels) == (7, 7)


def test_private_marker_requires_stable_seven_pixel_depth_window():
    module = _acceptance_module()
    segmentation = np.zeros((15, 15), dtype=np.int64)
    segmentation[5:10, 5:10] = 7
    labels = {7: "/World/scene_0/rpent_private_camera_marker/base_link/visuals"}

    with pytest.raises(RuntimeError, match="stable 7x7"):
        module._marker_interior_pixel(segmentation, labels)


def test_private_red_marker_mask_requires_dominant_stable_interior():
    module = _acceptance_module()
    rgb = np.zeros((15, 15, 3), dtype=np.uint8)
    rgb[3:12, 3:12, 0] = 220
    rgb[3:12, 3:12, 1] = 20
    rgb[3:12, 3:12, 2] = 20

    assert module._red_marker_interior_pixels(rgb)[0] == (7, 7)
    rgb[..., 1] = rgb[..., 0]
    assert module._red_marker_interior_pixels(rgb) == []


def test_private_marker_training_geometry_requires_two_affine_dimensions():
    module = _acceptance_module()

    assert module._point_set_rank([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) == 1
    assert module._point_set_rank([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) == 2


def test_radio_handle_grasp_frame_points_eef_down_and_opens_across_handle():
    module = _radio_acceptance_module()

    quat = module._top_down_grasp_quaternion(
        np.array([0.0, 0.0, 0.5]),
        np.array([0.2, 0.0, 0.5]),
    )
    x, y, z, w = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-8)
    np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def _write_radio_near_snapshot_manifest(tmp_path):
    snapshot = tmp_path / "radio_near_state.pt"
    snapshot.write_bytes(b"serialized-state")
    manifest = {
        "kind": "rpent_behavior_planner_restore",
        "task": {
            "control_mode": "planner_tools",
            "activity_instance_id": 211,
            "seed": 211,
        },
        "invariants": {
            "head_radio_visible": True,
            "attachments": {
                "left": {"backend": False, "simulator": False},
                "right": {"backend": False, "simulator": False},
            },
            "radio_toggled_on": False,
            "radio_navigation_error": {
                "position_m": 0.05,
                "orientation_rad": float(np.deg2rad(5.0)),
            },
        },
    }
    Path(f"{snapshot}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot, manifest


def test_radio_near_snapshot_gate_accepts_exact_boundary(tmp_path):
    module = _radio_acceptance_module()
    snapshot, manifest = _write_radio_near_snapshot_manifest(tmp_path)

    assert module._require_radio_near_snapshot_manifest(snapshot) == manifest


def test_cli_instance_and_seed_must_match_snapshot_manifest(tmp_path):
    module = _radio_acceptance_module()
    _snapshot, manifest = _write_radio_near_snapshot_manifest(tmp_path)

    module._require_cli_snapshot_identity(
        SimpleNamespace(activity_instance_id=211, seed=211), manifest
    )
    with pytest.raises(ValueError, match="CLI seed does not match"):
        module._require_cli_snapshot_identity(
            SimpleNamespace(activity_instance_id=211, seed=212), manifest
        )


def _write_bound_public_geometry_report(tmp_path, module, snapshot):
    target = [1.0, 2.0, 3.0]
    source = {
        "schema_version": 1,
        "operation": "inspect",
        "snapshot": {
            "sha256": module._sha256(snapshot),
            "manifest_sha256": module._sha256(Path(f"{snapshot}.manifest.json")),
        },
        "public_visual_selection": {
            "camera": "head",
            "u_column": 360,
            "v_row": 532,
            "selection_method": "manual_visual_review_of_real_radio_rgb",
            "private_truth_used_for_tool_target": False,
        },
        "public_target_xyz": target,
        "public_grasp_geometry": {
            "handle_points_world": [[0.9, 2.0, 3.0], [1.1, 2.0, 3.0]],
            "grasp_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "approach_vector": [0.0, 0.0, -1.0],
        },
        "public_tool_calls": [
            {
                "tool": "observe",
                "call_arguments": {"camera": "head"},
                "result": {
                    "primitive_success": True,
                    "frame_id": "head:7",
                },
            },
            {
                "tool": "pixel_to_world",
                "call_arguments": {
                    "camera": "head",
                    "frame_id": "head:7",
                    "u": 360,
                    "v": 532,
                    "output_frame": "world",
                },
                "result": {
                    "primitive_success": True,
                    "metrics": {"frame_id": "head:7"},
                    "diagnostics": {
                        "xyz": target,
                        "output_frame": "world",
                    },
                },
            },
        ],
    }
    path = tmp_path / "radio_tool_acceptance.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path, source


def test_legacy_geometry_reuse_requires_snapshot_and_public_transcript_binding(
    tmp_path,
):
    module = _radio_acceptance_module()
    snapshot, _manifest = _write_radio_near_snapshot_manifest(tmp_path)
    path, source = _write_bound_public_geometry_report(tmp_path, module, snapshot)

    target, grasp, evidence = module._load_bound_public_geometry(
        path, snapshot=snapshot
    )

    assert target == pytest.approx([1.0, 2.0, 3.0])
    assert grasp["grasp_quat_xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert evidence["transcript_validated"] is True

    source["snapshot"]["manifest_sha256"] = "wrong"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest_sha256 mismatch"):
        module._load_bound_public_geometry(path, snapshot=snapshot)

    source["snapshot"]["manifest_sha256"] = module._sha256(
        Path(f"{snapshot}.manifest.json")
    )
    source["public_tool_calls"][1]["call_arguments"]["u"] = 361
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"ordered observe\+world pixel_to_world"):
        module._load_bound_public_geometry(path, snapshot=snapshot)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda manifest: manifest.update(kind="wrong"), "wrong kind"),
        (
            lambda manifest: manifest["task"].update(control_mode="full_task_vla"),
            "control_mode mismatch",
        ),
        (
            lambda manifest: manifest["task"].update(activity_instance_id=210),
            "activity_instance_id mismatch",
        ),
        (lambda manifest: manifest["task"].update(seed=210), "seed mismatch"),
        (
            lambda manifest: manifest["invariants"].update(head_radio_visible=False),
            "head_radio_visible=true",
        ),
        (
            lambda manifest: manifest["invariants"]["attachments"]["left"].update(
                backend=True
            ),
            "left hand must have no attachment",
        ),
        (
            lambda manifest: manifest["invariants"]["attachments"]["right"].update(
                simulator=True
            ),
            "right hand must have no attachment",
        ),
        (
            lambda manifest: manifest["invariants"].update(radio_toggled_on=True),
            "radio_toggled_on=false",
        ),
        (
            lambda manifest: manifest["invariants"]["radio_navigation_error"].update(
                position_m=0.050001
            ),
            "position_m exceeds limit",
        ),
        (
            lambda manifest: manifest["invariants"]["radio_navigation_error"].update(
                orientation_rad=float(np.deg2rad(5.0)) + 1e-6
            ),
            "orientation_rad exceeds limit",
        ),
    ],
)
def test_radio_near_snapshot_gate_rejects_unsafe_manifest(tmp_path, mutation, error):
    module = _radio_acceptance_module()
    snapshot, manifest = _write_radio_near_snapshot_manifest(tmp_path)
    mutation(manifest)
    Path(f"{snapshot}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=error):
        module._require_radio_near_snapshot_manifest(snapshot)


def test_post_pick_visual_validation_flag_requires_pick_and_positive_timeout():
    module = _radio_acceptance_module()

    module._validate_post_pick_visual_validation_args(
        SimpleNamespace(
            post_pick_visual_validation=True,
            post_pick_visual_validation_timeout_s=12.0,
            operation="pick",
        )
    )
    with pytest.raises(ValueError, match="requires operation=pick"):
        module._validate_post_pick_visual_validation_args(
            SimpleNamespace(
                post_pick_visual_validation=True,
                post_pick_visual_validation_timeout_s=12.0,
                operation="inspect",
            )
        )
    with pytest.raises(ValueError, match="finite and positive"):
        module._validate_post_pick_visual_validation_args(
            SimpleNamespace(
                post_pick_visual_validation=True,
                post_pick_visual_validation_timeout_s=0.0,
                operation="pick",
            )
        )
    with pytest.raises(ValueError, match="three finite values"):
        module._validate_post_pick_visual_validation_args(
            SimpleNamespace(
                post_pick_visual_validation=True,
                post_pick_visual_validation_timeout_s=12.0,
                post_pick_visual_offset_world_m=[0.0, float("nan"), 0.2],
                operation="pick",
            )
        )


def test_post_pick_visual_validation_uses_public_offset_and_preserves_tool_order():
    module = _radio_acceptance_module()
    calls = []

    class _Env:
        _video_frames = 2

        @staticmethod
        def move_to(**kwargs):
            calls.append(kwargs)
            _Env._video_frames = 3
            return {"primitive_success": True, "stop_reason": "reached"}

    public_calls = [{"tool": "pick", "result": {"primitive_success": True}}]
    result, report = module._run_post_pick_visual_validation(
        _Env(),
        hand="left",
        public_target_xyz=[1.0, 2.0, 3.0],
        grasp_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        timeout_s=23.0,
        public_tool_calls=public_calls,
        phase_boundaries=[],
    )

    assert result["primitive_success"] is True
    assert [call["tool"] for call in public_calls] == ["pick", "move_to"]
    assert len(calls) == 1
    assert calls[0].pop("target_xyz") == pytest.approx([1.10, 1.83, 3.20])
    assert calls[0] == {
        "hand": "left",
        "frame": "world",
        "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "plan_only": False,
        "position_tolerance_m": 0.02,
        "orientation_tolerance_rad": 0.087,
        "timeout_s": 23.0,
    }
    assert report["target_xyz"] == pytest.approx([1.10, 1.83, 3.20])
    assert report["enabled"] is True
    assert report["target_derivation"] == {
        "source": "public_target_xyz",
        "offset_world_m": [0.10, -0.17, 0.20],
        "formula": "public_target_xyz + fixed_world_offset",
        "uses_private_truth": False,
        "uses_backend_pose": False,
    }


def test_post_pick_visual_validation_accepts_public_cli_offset():
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 2

        @staticmethod
        def move_to(**_kwargs):
            _Env._video_frames = 3
            return {"primitive_success": True, "stop_reason": "reached"}

    public_calls = []
    _result, report = module._run_post_pick_visual_validation(
        _Env(),
        hand="right",
        public_target_xyz=[1.0, 2.0, 3.0],
        grasp_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        timeout_s=23.0,
        public_tool_calls=public_calls,
        phase_boundaries=[],
        offset_world_m=[-0.06, 0.10, 0.20],
    )

    assert report["target_xyz"] == pytest.approx([0.94, 2.10, 3.20])
    assert report["offset_world_m"] == pytest.approx([-0.06, 0.10, 0.20])
    assert report["target_derivation"]["offset_world_m"] == pytest.approx(
        [-0.06, 0.10, 0.20]
    )
    assert public_calls[0]["target_derivation"]["uses_private_truth"] is False


def test_post_pick_visual_validation_call_precedes_private_audit():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)
    pick_branch = main_source.index('elif args.operation == "pick":')
    pick_tool = main_source.index('tool="pick"', pick_branch)

    assert pick_tool < main_source.index("_run_post_pick_visual_validation(")
    assert main_source.index("_run_post_pick_visual_validation(") < main_source.index(
        "_collect_posthoc_private_evidence("
    )
    helper_source = inspect.getsource(module._run_post_pick_visual_validation)
    assert "object_scope" not in helper_source
    assert ".backend" not in helper_source
    assert "get_eef_pose" not in helper_source


def _public_operation_args(operation, **overrides):
    values = {
        "operation": operation,
        "timeout_s": 90.0,
        "pick_timeout_s": None,
        "rotate_timeout_s": 45.0,
        "release_timeout_s": 30.0,
        "press_timeout_s": 60.0,
        "navigate_timeout_s": 90.0,
        "navigate_standoff_m": 1.0,
        "camera": "head",
        "reuse_public_geometry_json": None,
        "hand": "left",
        "button_camera": None,
        "button_u": None,
        "button_v": None,
        "button_depth_window_px": 7,
        "button_selection_source_json": None,
        "visual_observer_camera": None,
        "observer_look_offset_world_m": [0.0, 0.0, -0.10],
        "observer_camera_offset_world_m": [0.0, -0.45, 0.02],
        "observer_move_timeout_s": 45.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_public_operation_argument_validation_is_fail_closed():
    module = _radio_acceptance_module()

    module._validate_public_operation_args(_public_operation_args("held_chain"))
    module._validate_public_operation_args(
        _public_operation_args("press_preview", button_camera="left_wrist")
    )
    module._validate_public_operation_args(
        _public_operation_args(
            "press",
            hand="right",
            button_camera="right_wrist",
            button_u=12,
            button_v=34,
            button_selection_source_json="preview.json",
        )
    )
    module._validate_public_operation_args(_public_operation_args("navigate"))

    with pytest.raises(ValueError, match="requires both --button-u and --button-v"):
        module._validate_public_operation_args(
            _public_operation_args(
                "press",
                button_camera="left_wrist",
                button_u=12,
                button_selection_source_json="preview.json",
            )
        )
    with pytest.raises(ValueError, match="requires --button-camera left_wrist"):
        module._validate_public_operation_args(_public_operation_args("press_preview"))
    with pytest.raises(ValueError, match="fresh head RGB-D"):
        module._validate_public_operation_args(
            _public_operation_args("navigate", reuse_public_geometry_json="prior.json")
        )
    with pytest.raises(ValueError, match="rotate timeout"):
        module._validate_public_operation_args(
            _public_operation_args("held_chain", rotate_timeout_s=float("nan"))
        )
    with pytest.raises(ValueError, match="requires camera=head"):
        module._validate_public_operation_args(
            _public_operation_args("navigate", camera="right_wrist")
        )


@pytest.mark.parametrize(
    "operation",
    ["held_chain", "press_preview", "press", "navigate", "observer_preview"],
)
def test_each_new_operation_forbids_reused_geometry(operation):
    module = _radio_acceptance_module()
    overrides = {"reuse_public_geometry_json": "prior.json"}
    if operation in {"press_preview", "press"}:
        overrides["button_camera"] = "left_wrist"
    if operation == "press":
        overrides.update(
            button_u=12,
            button_v=34,
            button_selection_source_json="preview.json",
        )
    if operation == "observer_preview":
        overrides["visual_observer_camera"] = "right_wrist"

    with pytest.raises(ValueError, match="fresh head RGB-D"):
        module._validate_public_operation_args(
            _public_operation_args(operation, **overrides)
        )


@pytest.mark.parametrize(
    ("hand", "observer_camera"),
    [("right", "left_wrist"), ("left", "right_wrist")],
)
def test_observer_preview_requires_fresh_head_and_opposite_wrist(hand, observer_camera):
    module = _radio_acceptance_module()
    valid = _public_operation_args(
        "observer_preview",
        hand=hand,
        visual_observer_camera=observer_camera,
    )

    module._validate_public_operation_args(valid)
    with pytest.raises(ValueError, match="requires --visual-observer-camera"):
        module._validate_public_operation_args(
            _public_operation_args("observer_preview", hand=hand)
        )
    with pytest.raises(ValueError, match="opposite wrist"):
        module._validate_public_operation_args(
            _public_operation_args(
                "observer_preview",
                hand=hand,
                visual_observer_camera=f"{hand}_wrist",
            )
        )
    with pytest.raises(ValueError, match="fresh head RGB-D"):
        module._validate_public_operation_args(
            _public_operation_args(
                "observer_preview",
                hand=hand,
                visual_observer_camera=observer_camera,
                reuse_public_geometry_json="prior.json",
            )
        )


def test_pick_observer_requires_fresh_head_and_rejects_nonfinite_offsets():
    module = _radio_acceptance_module()

    module._validate_public_operation_args(
        _public_operation_args(
            "pick", hand="right", visual_observer_camera="left_wrist"
        )
    )
    with pytest.raises(ValueError, match="fresh head RGB-D"):
        module._validate_public_operation_args(
            _public_operation_args(
                "pick",
                hand="right",
                visual_observer_camera="left_wrist",
                reuse_public_geometry_json="prior.json",
            )
        )
    with pytest.raises(ValueError, match="observer look offset"):
        module._validate_public_operation_args(
            _public_operation_args(
                "pick",
                hand="right",
                visual_observer_camera="left_wrist",
                observer_look_offset_world_m=[0.0, float("nan"), -0.10],
            )
        )


@pytest.mark.parametrize("invalid_timeout", [0.0, -1.0, float("nan")])
def test_observer_move_timeout_must_be_positive_and_finite(invalid_timeout):
    module = _radio_acceptance_module()

    with pytest.raises(ValueError, match="observer move timeout"):
        module._validate_public_operation_args(
            _public_operation_args(
                "observer_preview",
                hand="right",
                visual_observer_camera="left_wrist",
                observer_move_timeout_s=invalid_timeout,
            )
        )


def test_observer_cli_defaults_and_direct_script_import_are_reliable(
    monkeypatch, tmp_path
):
    module = _radio_acceptance_module()
    script = Path(module.__file__).resolve()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--activity-instance-dir",
            "instances",
            "--snapshot",
            "state.pt",
            "--output-dir",
            "artifacts",
            "--operation",
            "observer_preview",
            "--visual-observer-camera",
            "left_wrist",
        ],
    )

    args = module._args()

    assert args.visual_observer_camera == "left_wrist"
    assert args.observer_look_offset_world_m == pytest.approx([0.0, 0.0, -0.10])
    assert args.observer_camera_offset_world_m == pytest.approx([0.0, -0.45, 0.02])
    assert args.observer_move_timeout_s == pytest.approx(45.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [*sys.argv, "--observer-move-timeout-s", "90"],
    )
    retry_args = module._args()
    module._validate_public_operation_args(retry_args)
    assert retry_args.observer_move_timeout_s == pytest.approx(90.0)

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "observer_preview" in completed.stdout
    assert "--visual-observer-camera" in completed.stdout


def test_opposite_wrist_observer_setup_is_public_schema_valid_and_phase_bounded(
    monkeypatch, tmp_path
):
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 5

        def move_to(self, **_kwargs):
            self._video_frames = 9
            return {"primitive_success": True, "stop_reason": "reached"}

        @staticmethod
        def observe(*, camera):
            return {
                "primitive_success": True,
                "stop_reason": "observed",
                "camera": camera,
                "frame_id": f"{camera}:12",
            }

    monkeypatch.setattr(
        module, "_save_public_rgbd_frame", lambda *_args, **_kwargs: None
    )
    public_calls = []
    phases = []

    move_result, observed, report = module._run_opposite_wrist_observer_setup(
        _Env(),
        action_hand="right",
        observer_camera="left_wrist",
        public_surface_xyz=[1.0, 2.0, 0.8],
        look_offset_world_m=[0.0, 0.0, -0.10],
        camera_offset_world_m=[0.0, -0.45, 0.02],
        timeout_s=90.0,
        output_dir=tmp_path,
        public_tool_calls=public_calls,
        phase_boundaries=phases,
    )

    assert move_result["primitive_success"] is True
    assert observed["frame_id"] == "left_wrist:12"
    assert report["action_hand"] == "right"
    assert report["observer_hand"] == "left"
    assert report["observer_camera"] == "left_wrist"
    assert report["geometry"]["surface_target_xyz"] == pytest.approx([1.0, 2.0, 0.8])
    assert report["geometry"]["look_offset_world_m"] == pytest.approx([0.0, 0.0, -0.10])
    assert report["geometry"]["camera_offset_world_m"] == pytest.approx(
        [0.0, -0.45, 0.02]
    )
    assert report["provenance"]["uses_private_truth"] is False
    assert [call["tool"] for call in public_calls] == ["move_to", "observe"]
    move_arguments = public_calls[0]["call_arguments"]
    assert move_arguments["hand"] == "left"
    assert move_arguments["frame"] == "world"
    assert move_arguments["target_xyz"] == pytest.approx(
        report["geometry"]["eef_position_world_xyz"]
    )
    assert move_arguments["target_quat_xyzw"] == pytest.approx(
        report["geometry"]["eef_quat_xyzw"]
    )
    assert move_arguments["timeout_s"] == pytest.approx(90.0)
    assert report["move_timeout_s"] == pytest.approx(90.0)
    for call in public_calls:
        schema = PLANNER_TOOL_SPECS[call["tool"]]["input_schema"]
        assert set(schema["required"]) <= set(call["call_arguments"])
        assert set(call["call_arguments"]) <= set(schema["properties"])
    assert phases == [
        {
            "sequence": 0,
            "tool": "move_to",
            "tool_role": "setup",
            "before_video_frame": 5,
            "after_video_frame": 9,
            "outcome": "returned",
        },
        {
            "sequence": 1,
            "tool": "observe",
            "tool_role": "targeting",
            "before_video_frame": 9,
            "after_video_frame": 9,
            "outcome": "returned",
        },
    ]
    helper_source = inspect.getsource(module._run_opposite_wrist_observer_setup)
    assert "object_scope" not in helper_source
    assert ".backend" not in helper_source
    assert "get_eef_pose" not in helper_source


def test_pick_with_observer_setup_runs_observer_before_phase_bounded_pick():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)
    observer_setup = main_source.index("_run_opposite_wrist_observer_setup(")
    pick_branch = main_source.index('elif args.operation == "pick":')
    pick_tool = main_source.index('tool="pick"', pick_branch)

    assert observer_setup < pick_tool
    assert 'tool_role="subject"' in main_source[pick_branch : pick_tool + 200]
    assert '"observer_non_active_arm_gripper_joints_locked"' in main_source
    assert '"shared_trunk_remains_active": True' in main_source
    assert '"observer_wrist_camera_world_pose_may_drift"' in main_source
    assert '"fixed_observer_world_pose_claimed": False' in main_source
    assert "object_scope" not in main_source[observer_setup:pick_tool]


def test_observer_locking_report_never_claims_fixed_world_pose():
    module = _radio_acceptance_module()

    report = module._observer_locking_report(
        action_hand="right",
        observer_hand="left",
        operation="pick",
    )

    assert report == {
        "executor_lock_scope": "non_active_arm_and_gripper_joints_only",
        "during_observer_move_non_active_arm_gripper": "right",
        "during_pick_non_active_arm_gripper": "left",
        "shared_trunk_remains_active": True,
        "observer_wrist_camera_world_pose_may_drift_during_pick": True,
        "fixed_observer_world_pose_claimed": False,
    }


def test_pick_timeout_flag_is_independent_with_legacy_fallback():
    module = _radio_acceptance_module()

    assert module._effective_pick_timeout_s(
        _public_operation_args("held_chain", timeout_s=111.0)
    ) == pytest.approx(111.0)
    assert module._effective_pick_timeout_s(
        _public_operation_args("held_chain", timeout_s=111.0, pick_timeout_s=222.0)
    ) == pytest.approx(222.0)


def test_press_requires_matching_reviewed_press_preview_selection(tmp_path):
    module = _radio_acceptance_module()
    snapshot, _manifest = _write_radio_near_snapshot_manifest(tmp_path)
    setup_target = [1.0, 2.0, 3.2]
    source = {
        "operation": "press_preview",
        "hand": "left",
        "execution_status": "runtime_press_preview_complete",
        "snapshot": {
            "sha256": module._sha256(snapshot),
            "manifest_sha256": module._sha256(Path(f"{snapshot}.manifest.json")),
        },
        "press_view_setup": {
            "button_camera": "left_wrist",
            "target_xyz": setup_target,
        },
        "press_preview_policy": {
            "visual_candidate_only": True,
            "automatically_authorizes_press": False,
        },
    }
    source_path = tmp_path / "radio_tool_acceptance.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    decision = {
        "accepted": True,
        "visual_findings": {"button_visible": True},
        "reviewed_button_selection": {
            "hand": "left",
            "camera": "left_wrist",
            "u_column": 101,
            "v_row": 202,
        },
    }
    (tmp_path / "leader_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )

    evidence = module._load_reviewed_button_selection(
        source_path,
        snapshot=snapshot,
        hand="left",
        camera="left_wrist",
        setup_target_xyz=setup_target,
        cli_u=101,
        cli_v=202,
    )

    assert evidence["button_visible"] is True
    assert evidence["reviewed_button_selection"]["u_column"] == 101
    with pytest.raises(RuntimeError, match="does not match CLI"):
        module._load_reviewed_button_selection(
            source_path,
            snapshot=snapshot,
            hand="left",
            camera="left_wrist",
            setup_target_xyz=setup_target,
            cli_u=102,
            cli_v=202,
        )


def test_public_tool_boundary_records_role_and_continuous_video_frames():
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 7

        def rotate_wrist(self, **_kwargs):
            self._video_frames = 11
            return {"primitive_success": True, "stop_reason": "reached"}

    env = _Env()
    calls = []
    phases = []
    result = module._call_public_tool_with_boundary(
        env,
        tool="rotate_wrist",
        tool_role="subject",
        call_kwargs={
            "hand": "right",
            "relative_axis_angle": [0.0, 0.0, 1.0, np.deg2rad(15.0)],
            "frame": "eef",
            "timeout_s": 45.0,
        },
        public_tool_calls=calls,
        phase_boundaries=phases,
    )

    assert result["primitive_success"] is True
    assert phases == [
        {
            "sequence": 0,
            "tool": "rotate_wrist",
            "tool_role": "subject",
            "before_video_frame": 7,
            "after_video_frame": 11,
            "outcome": "returned",
        }
    ]
    assert calls[0]["phase_boundary"] == phases[0]
    assert calls[0]["tool_role"] == "subject"


def test_public_tool_boundary_records_exception_before_reraising():
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 3

        def release(self, **_kwargs):
            self._video_frames = 4
            raise TimeoutError("bounded release timeout")

    calls = []
    phases = []
    with pytest.raises(TimeoutError, match="bounded release timeout"):
        module._call_public_tool_with_boundary(
            _Env(),
            tool="release",
            tool_role="subject",
            call_kwargs={"hand": "right"},
            public_tool_calls=calls,
            phase_boundaries=phases,
        )

    assert phases[0]["before_video_frame"] == 3
    assert phases[0]["after_video_frame"] == 4
    assert phases[0]["outcome"] == "exception"
    assert calls[0]["error"] == "TimeoutError: bounded release timeout"


def test_motion_tool_boundary_requires_video_frame_advance():
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 5

        @staticmethod
        def move_to(**_kwargs):
            return {"primitive_success": True, "stop_reason": "reached"}

    calls = []
    phases = []
    with pytest.raises(RuntimeError, match="did not advance"):
        module._call_public_tool_with_boundary(
            _Env(),
            tool="move_to",
            tool_role="setup",
            call_kwargs={"hand": "left", "target_xyz": [1.0, 2.0, 3.0]},
            public_tool_calls=calls,
            phase_boundaries=phases,
        )

    assert phases[0]["outcome"] == "video_frame_not_advanced"
    assert calls[0]["result"]["primitive_success"] is True


def test_failure_health_probe_calls_public_observe_once_without_video_counter():
    module = _radio_acceptance_module()

    class _Env:
        calls = 0

        def observe(self, *, camera):
            self.calls += 1
            assert camera == "head"
            return {
                "primitive_success": True,
                "stop_reason": "observed",
                "frame_id": "head:health",
            }

    env = _Env()
    public_calls = []
    result = module._probe_public_env_health_after_failure(
        env,
        public_tool_calls=public_calls,
        phase_boundaries=[],
    )

    assert env.calls == 1
    assert result == {
        "attempted": True,
        "responsive": True,
        "primitive_success": True,
        "stop_reason": "observed",
    }
    assert public_calls[0]["tool"] == "observe"
    assert public_calls[0]["tool_role"] == "cleanup"
    assert "phase_boundary" not in public_calls[0]


def test_failure_health_probe_preserves_failed_observe_as_evidence():
    module = _radio_acceptance_module()

    class _Env:
        _video_frames = 4
        calls = 0

        def observe(self, *, camera):
            self.calls += 1
            self._video_frames += 1
            raise TimeoutError(f"{camera} health timeout")

    env = _Env()
    public_calls = []
    phases = []
    result = module._probe_public_env_health_after_failure(
        env,
        public_tool_calls=public_calls,
        phase_boundaries=phases,
    )

    assert env.calls == 1
    assert result["attempted"] is True
    assert result["responsive"] is False
    assert result["error"] == "TimeoutError: head health timeout"
    assert public_calls[0]["tool_role"] == "cleanup"
    assert phases[0]["outcome"] == "exception"


def test_public_chain_gates_on_primitive_success_not_task_success():
    module = _radio_acceptance_module()

    module._require_primitive_success(
        {"primitive_success": True, "task_success": False}, tool="subject"
    )
    with pytest.raises(RuntimeError, match="subject failed: collision"):
        module._require_primitive_success(
            {
                "primitive_success": False,
                "task_success": True,
                "stop_reason": "collision",
            },
            tool="subject",
        )


def test_held_chain_uses_pick_setup_then_rotate_and_release_subjects():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)
    held_source = main_source[
        main_source.index('elif args.operation == "held_chain"') : main_source.index(
            'elif args.operation in {"press_preview", "press"}'
        )
    ]

    assert held_source.index('tool="pick"') < held_source.index('tool="rotate_wrist"')
    assert held_source.index('tool="rotate_wrist"') < held_source.index(
        'tool="release"'
    )
    assert 'tool_role="setup"' in held_source
    assert held_source.count('tool_role="subject"') == 2
    assert '"relative_axis_angle": list(HELD_ROTATION_AXIS_ANGLE_RAD)' in held_source
    assert '"frame": "eef"' in held_source
    assert '"opening": 1.0' in held_source
    assert '"retreat_vector": [0.0, 0.0, 1.0]' in held_source
    assert '"retreat_m": 0.03' in held_source
    assert "start_video_segment" not in held_source


def test_held_axis_angle_passes_real_schema_client_and_executor_validator():
    module = _radio_acceptance_module()
    axis_angle = list(module.HELD_ROTATION_AXIS_ANGLE_RAD)
    schema = PLANNER_TOOL_SPECS["rotate_wrist"]["input_schema"]["properties"][
        "relative_axis_angle"
    ]

    assert len(axis_angle) == schema["minItems"] == schema["maxItems"] == 4
    quat = _axis_angle_to_quat_xyzw(axis_angle)
    assert quat == pytest.approx(
        [0.0, 0.0, np.sin(np.deg2rad(7.5)), np.cos(np.deg2rad(7.5))]
    )

    class _Rpc:
        recorded = None

        def call(self, method, *, kwargs=None, timeout_s=None):
            del timeout_s
            if method == "env.get_env_meta":
                return {"mode": "planner_tools"}
            self.recorded = (method, kwargs)
            return {"primitive_success": True}

    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"mode": "planner_tools"})
    result = client.rotate_wrist(
        hand="right",
        relative_axis_angle=axis_angle,
        frame="eef",
        timeout_s=45.0,
    )

    assert result["primitive_success"] is True
    assert rpc.recorded == (
        "env.rotate_wrist",
        {
            "hand": "right",
            "target_quat_xyzw": None,
            "relative_axis_angle": axis_angle,
            "frame": "eef",
            "timeout_s": 45.0,
        },
    )


def test_press_direction_is_negative_normalized_public_surface_normal():
    module = _radio_acceptance_module()
    projection = {
        "diagnostics": {
            "surface_normal": [0.0, 3.0, 4.0],
            "output_frame": "world",
        }
    }

    assert module._negative_public_surface_normal(projection) == pytest.approx(
        [0.0, -0.6, -0.8]
    )
    with pytest.raises(RuntimeError, match="surface normal is invalid"):
        module._negative_public_surface_normal(
            {
                "diagnostics": {
                    "surface_normal": [0.0, 0.0, 0.0],
                    "output_frame": "world",
                }
            }
        )
    with pytest.raises(RuntimeError, match="must be in world frame"):
        module._negative_public_surface_normal(
            {
                "diagnostics": {
                    "surface_normal": [0.0, 0.0, 1.0],
                    "output_frame": "camera",
                }
            }
        )


def test_press_and_navigate_targets_are_public_and_phase_bounded():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)
    press_source = main_source[
        main_source.index(
            'elif args.operation in {"press_preview", "press"}'
        ) : main_source.index('elif args.operation == "navigate"')
    ]
    navigate_source = main_source[
        main_source.index('elif args.operation == "navigate"') : main_source.index(
            'if args.operation == "preview_pick"'
        )
    ]

    assert "PRESS_VIEW_OFFSET_WORLD_M" in press_source
    assert 'tool="observe"' in press_source
    assert 'tool="pixel_to_world"' in press_source
    assert 'tool="press"' in press_source
    assert "_negative_public_surface_normal(button_projection)" in press_source
    assert "object_scope" not in press_source
    assert ".backend" not in press_source
    assert "get_eef_pose" not in press_source
    assert 'tool="navigate_to"' in navigate_source
    assert navigate_source.index('tool="navigate_to"') < navigate_source.index(
        'metadata={"visual_role": "post_navigation_radio_reobserve"}'
    )
    assert navigate_source.index(
        'metadata={"visual_role": "post_navigation_radio_reobserve"}'
    ) < navigate_source.index(
        'metadata={"visual_role": "post_navigation_radio_reprojection"}'
    )
    assert '"camera": "head"' in navigate_source
    assert '"output_frame": "world"' in navigate_source
    assert '"source": "fresh_head_rgbd_pixel_to_world"' in navigate_source
    assert "start_video_segment" not in press_source + navigate_source


def test_radio_visual_review_artifacts_record_frames_duration_and_hashes(tmp_path):
    module = _radio_acceptance_module()
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"finalized-mp4-placeholder")

    def mosaic(seed: int) -> np.ndarray:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        frame[:4, :5] = [seed, 10, 20]
        frame[:4, 5:] = [30, seed, 40]
        frame[4:, :5] = [50, 60, seed]
        return frame

    metadata = module._write_visual_review_artifacts(
        video_path=video_path,
        output_dir=tmp_path,
        phase_frames={
            "first": (0, mosaic(1)),
            "middle": (4, mosaic(2)),
            "last": (8, mosaic(3)),
        },
        frame_count=9,
        fps=3.0,
    )

    assert metadata["frames"] == 9
    assert metadata["duration_s"] == pytest.approx(3.0)
    assert metadata["sha256"] == module._sha256(video_path)
    assert metadata["visual_review_required"] is True
    assert metadata["visual_decision"] is None
    assert metadata["automatic_visual_success_declared"] is False
    assert [
        metadata["extracted_phases"][phase]["frame_index"]
        for phase in ("first", "middle", "last")
    ] == [0, 4, 8]
    for phase in ("first", "middle", "last"):
        phase_record = metadata["extracted_phases"][phase]
        assert Path(phase_record["mosaic"]["path"]).is_file()
        for camera in ("head", "left_wrist", "right_wrist"):
            artifact = phase_record["cameras"][camera]
            artifact_path = Path(artifact["path"])
            assert artifact_path.is_file()
            assert artifact["sha256"] == module._sha256(artifact_path)
    contact_sheet = imageio.imread(metadata["contact_sheet"]["path"])
    assert contact_sheet.shape == (12, 15, 3)


def test_phase_boundary_extraction_writes_before_mid_after_all_cameras(tmp_path):
    module = _radio_acceptance_module()
    video_path = tmp_path / "episode.mp4"
    writer = imageio.get_writer(video_path, fps=15)
    try:
        for value in range(6):
            frame = np.full((32, 32, 3), value * 20, dtype=np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()
    boundary = {
        "sequence": 0,
        "tool": "move_to",
        "tool_role": "setup",
        "before_video_frame": 1,
        "after_video_frame": 5,
        "outcome": "returned",
    }

    metadata = module._extract_visual_review_artifacts(
        video_path,
        tmp_path,
        phase_boundaries=[boundary],
    )

    extracted = metadata["phase_boundary_frames"][0]
    assert {
        phase: extracted["phases"][phase]["frame_index"]
        for phase in ("before", "mid", "after")
    } == {"before": 0, "mid": 2, "after": 4}
    for phase in ("before", "mid", "after"):
        for camera in ("head", "left_wrist", "right_wrist"):
            artifact = extracted["phases"][phase]["cameras"][camera]
            path = Path(artifact["path"])
            assert path.is_file()
            assert artifact["sha256"] == module._sha256(path)

    with pytest.raises(RuntimeError, match="outside the finalized MP4"):
        module._phase_boundary_frame_indices(
            [{**boundary, "after_video_frame": 7}], frame_count=6
        )


def test_radio_private_truth_is_posthoc_and_cannot_gate_public_tool_execution(
    tmp_path,
):
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)
    posthoc_call = main_source.index("_collect_posthoc_private_evidence(")
    pre_posthoc_source = main_source[:posthoc_call]

    assert main_source.index("env.pixel_to_world(") < posthoc_call
    assert main_source.index("action_result = env.move_to(") < posthoc_call
    pick_branch = main_source.index('elif args.operation == "pick":')
    assert main_source.index('tool="pick"', pick_branch) < posthoc_call
    assert "object_scope" not in pre_posthoc_source
    assert "_private_audit(" not in pre_posthoc_source
    assert "_radio_mask(" not in pre_posthoc_source
    assert "pick did not attach" not in main_source
    assert 'report["status"] = "failed"' not in main_source
    assert '"status": "pending_visual_review"' in main_source

    missing_radio_env = SimpleNamespace(
        _env=SimpleNamespace(
            omnigibson_env=SimpleNamespace(task=SimpleNamespace(object_scope={}))
        )
    )
    evidence = module._collect_posthoc_private_evidence(
        missing_radio_env,
        args=SimpleNamespace(
            camera="head",
            u=1,
            v=2,
            handle_u1=3,
            handle_v1=4,
            handle_u2=5,
            handle_v2=6,
        ),
        action_dir=tmp_path,
        source_snapshot=tmp_path / "state.pt",
    )
    assert evidence["available"] is False
    assert evidence["does_not_gate_execution"] is True
    assert evidence["phase"] == "after_all_public_tool_calls"
    assert "missing during posthoc audit" in evidence["error"]


def test_radio_mp4_extraction_runs_after_video_is_closed():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)

    assert main_source.index("env.close()") < main_source.index(
        "_extract_visual_review_artifacts("
    )
    assert 'report["status"] = "pending_visual_review"' in main_source


def test_execution_failure_status_and_original_error_survive_cleanup():
    module = _radio_acceptance_module()
    main_source = inspect.getsource(module.main)

    except_source = main_source[
        main_source.index("    except Exception as exc:") : main_source.index(
            "    finally:"
        )
    ]
    finally_source = main_source[main_source.index("    finally:") :]
    assert 'report["status"] = "execution_failed"' in except_source
    assert 'report["error"] = f"{type(exc).__name__}: {exc}"' in except_source
    assert "_probe_public_env_health_after_failure(" in except_source
    assert "raise" in except_source
    assert 'if report.get("status") != "execution_failed":' in finally_source
