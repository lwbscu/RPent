from __future__ import annotations

import base64
import json
import threading
from types import SimpleNamespace

from mcp import types

from robots.behavior.mcp_server import (
    _ToolCallAdmission,
    _toolkit_to_mcp_content,
)
from robots.behavior.toolkit import BehaviorToolkit, BehaviorToolResult


def _public_result(**extra):
    return {
        "primitive_success": True,
        "task_success": False,
        "official_success_source": 'info["done"]["success"]',
        "stop_reason": "observed",
        **extra,
    }


def _toolkit(tmp_path, env) -> BehaviorToolkit:
    return BehaviorToolkit(
        primitives_kwargs={
            "env": env,
            "model": object(),
            "output_dir": tmp_path,
            "initial_info": {"done": {"success": False}},
        }
    )


def _mcp_content(result):
    return _toolkit_to_mcp_content(result)


def test_concurrent_mcp_waiter_reuses_first_verified_success_without_readmission():
    first_entered = threading.Event()
    release_first = threading.Event()
    terminal = SimpleNamespace(
        is_finish=True,
        result={
            "_finish": True,
            "task_success": True,
            "official_success_receipt": {"env_step": 7},
        },
    )

    class _Toolkit:
        calls = 0

        def execute_tool(self, _name, _arguments):
            self.calls += 1
            first_entered.set()
            assert release_first.wait(timeout=2)
            return terminal

    toolkit = _Toolkit()
    admission = _ToolCallAdmission(toolkit)
    results = []
    first = threading.Thread(
        target=lambda: results.append(admission.execute("observe", {}))
    )
    second = threading.Thread(
        target=lambda: results.append(admission.execute("navigate_to", {}))
    )
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert toolkit.calls == 1
    assert results == [terminal, terminal]


def test_observe_public_result_reaches_mcp_as_image_blocks(tmp_path):
    rgb = b"\x89PNG\r\n\x1a\nhead-rgb"
    depth = b"\x89PNG\r\n\x1a\nhead-depth"
    env = SimpleNamespace(
        observe=lambda *, camera: _public_result(
            camera=camera,
            frame_id=f"{camera}:0:test",
            _image_bytes=rgb,
            _depth_image_bytes=depth,
        )
    )

    result = _toolkit(tmp_path, env).execute_tool("observe", {"camera": "head"})
    public = result.result
    content, is_error = _mcp_content(result)

    assert isinstance(public["_image_bytes"], bytes)
    assert isinstance(public["_depth_image_bytes"], bytes)
    assert is_error is False
    assert [type(block) for block in content] == [
        types.TextContent,
        types.ImageContent,
        types.ImageContent,
    ]
    assert [base64.b64decode(block.data) for block in content[1:]] == [rgb, depth]
    text = json.loads(content[0].text)
    assert "_image_bytes" not in text
    assert "_depth_image_bytes" not in text
    assert text["camera"] == "head"
    assert [item["kind"] for item in text["mcp_image_block_order"]] == [
        "rgb",
        "depth_visualization",
    ]


def test_observe_depth_probe_reaches_llm_as_text_plus_same_rgbd_blocks(tmp_path):
    rgb = b"\x89PNG\r\n\x1a\nleft-wrist-rgb"
    depth = b"\x89PNG\r\n\x1a\nleft-wrist-depth"
    requested_probe = {
        "frame_id": "left_wrist:4:selected-target",
        "u": 120,
        "v": 80,
        "depth_window_px": 7,
        "assessment": "target_point_visually_confirmed",
    }
    public_probe = {
        "source": "llm_selected_pixel",
        "measurement": "first_visible_surface_at_llm_selected_pixel",
        "pixel": {"u": 120, "v": 80},
        "depth_window_px": 7,
        "optical_axis_depth_m": 0.184,
        "camera_range_m": 0.187,
        "target_point_camera_xyz_m": [0.011, -0.006, -0.184],
        "target_to_palm_m": 0.134,
        "target_to_grip_point_m": 0.079,
        "target_to_finger_roots_m": 0.071,
        "quality": {
            "mad_m": 0.001,
            "valid_ratio": 1.0,
            "cluster_ratio": 0.88,
            "sample_count": 49,
            "valid_count": 49,
            "cluster_count": 43,
            "confidence": 0.86,
        },
        "semantic_target_verified": False,
        "motion_authorization": False,
        "hand_geometry": {
            "available": True,
            "resolved_hand": "left",
            "source": "frame_bound_live_r1pro_link_transforms",
            "target_to_finger_roots_individual_m": [0.071, 0.074],
            "geometry_sha256": "b" * 64,
            "frame_id": "left_wrist:4:selected-target",
            "capture_group_id": "capture:4:current",
            "env_step": 4,
            "target_point_camera_frame": "effective_usd_camera",
            "camera_axes": "+X right,+Y up,-Z forward",
            "distance_computation_frame": "world",
            "guidance_only": True,
            "semantic_target_verified": False,
            "collision_authorization": False,
            "close_authorization": False,
            "open_authorization": False,
        },
        "lineage": {
            "camera": "left_wrist",
            "frame_id": "left_wrist:4:selected-target",
            "capture_group_id": "capture:4:current",
            "env_step": 4,
            "receipt_sha256": "a" * 64,
        },
    }
    calls = []

    def observe(**kwargs):
        calls.append(kwargs)
        return _public_result(
            camera=kwargs["camera"],
            frame_id=requested_probe["frame_id"],
            depth_probe=public_probe,
            _image_bytes=rgb,
            _depth_image_bytes=depth,
        )

    result = _toolkit(tmp_path, SimpleNamespace(observe=observe)).execute_tool(
        "observe",
        {"camera": "left_wrist", "depth_probe": requested_probe},
    )
    content, is_error = _mcp_content(result)

    assert calls == [
        {
            "camera": "left_wrist",
            "depth_probe": requested_probe,
        }
    ]
    assert is_error is False
    assert [type(block) for block in content] == [
        types.TextContent,
        types.ImageContent,
        types.ImageContent,
    ]
    assert [base64.b64decode(block.data) for block in content[1:]] == [rgb, depth]
    text = json.loads(content[0].text)
    assert text["depth_probe"] == public_probe
    assert text["depth_probe"]["semantic_target_verified"] is False
    assert text["depth_probe"]["motion_authorization"] is False
    assert text["depth_probe"]["target_point_camera_xyz_m"] == [
        0.011,
        -0.006,
        -0.184,
    ]
    assert text["depth_probe"]["hand_geometry"]["guidance_only"] is True
    assert text["depth_probe"]["target_to_finger_roots_m"] == min(
        text["depth_probe"]["hand_geometry"]["target_to_finger_roots_individual_m"]
    )
    for forbidden in (
        "target_point_world_xyz_m",
        "surface_normal",
        "projection_id",
    ):
        assert forbidden not in text["depth_probe"]
    assert "_image_bytes" not in content[0].text
    assert "_depth_image_bytes" not in content[0].text


def test_visual_checkpoint_multicamera_images_do_not_leak_into_text(tmp_path):
    cameras = ("head", "left_wrist", "right_wrist")
    expected_images = []
    images = {}
    for camera in cameras:
        rgb = b"\x89PNG\r\n\x1a\n" + f"{camera}-rgb".encode()
        depth = b"\x89PNG\r\n\x1a\n" + f"{camera}-depth".encode()
        expected_images.extend((rgb, depth))
        images[camera] = {
            "camera": camera,
            "frame_id": f"{camera}:0:test",
            "_image_bytes": rgb,
            "_depth_image_bytes": depth,
        }
    env = SimpleNamespace(
        save_robot_state_checkpoint=lambda *, semantic_label: _public_result(
            stop_reason="saved_visual_checkpoint",
            semantic_label=semantic_label,
            images=images,
        )
    )

    result = _toolkit(tmp_path, env).execute_tool(
        "save_robot_state_checkpoint",
        {"semantic_label": "initial visual anchor"},
    )
    public = result.result
    content, is_error = _mcp_content(result)

    for camera in cameras:
        assert isinstance(public["images"][camera]["_image_bytes"], bytes)
        assert isinstance(public["images"][camera]["_depth_image_bytes"], bytes)
    assert is_error is False
    assert isinstance(content[0], types.TextContent)
    assert all(isinstance(block, types.ImageContent) for block in content[1:])
    assert len(content) == 7
    assert [base64.b64decode(block.data) for block in content[1:]] == expected_images
    text = json.loads(content[0].text)
    assert set(text["images"]) == set(cameras)
    assert all(
        text["images"][camera]["frame_id"] == f"{camera}:0:test" for camera in cameras
    )
    assert text["mcp_image_block_order"] == [
        {
            "content_block_index": 1,
            "camera": "head",
            "kind": "rgb",
        },
        {
            "content_block_index": 2,
            "camera": "head",
            "kind": "depth_visualization",
        },
        {
            "content_block_index": 3,
            "camera": "left_wrist",
            "kind": "rgb",
        },
        {
            "content_block_index": 4,
            "camera": "left_wrist",
            "kind": "depth_visualization",
        },
        {
            "content_block_index": 5,
            "camera": "right_wrist",
            "kind": "rgb",
        },
        {
            "content_block_index": 6,
            "camera": "right_wrist",
            "kind": "depth_visualization",
        },
    ]
    assert "_image_bytes" not in content[0].text
    assert "_depth_image_bytes" not in content[0].text
    assert "\"b'" not in content[0].text


def test_plain_json_tool_result_remains_one_serializable_text_block():
    payload = {
        "primitive_success": True,
        "task_success": False,
        "nested": {"items": [1, "two", False, None]},
    }

    content, is_error = _mcp_content(BehaviorToolResult(name="plain", result=payload))

    assert is_error is False
    assert len(content) == 1
    assert isinstance(content[0], types.TextContent)
    assert json.loads(content[0].text) == payload


def test_invalid_behavior_image_carrier_becomes_structured_mcp_error(tmp_path):
    env = SimpleNamespace(
        observe=lambda *, camera: _public_result(
            camera=camera,
            frame_id=f"{camera}:0:test",
            _image_bytes="not-bytes",
        )
    )

    result = _toolkit(tmp_path, env).execute_tool("observe", {"camera": "head"})
    content, is_error = _mcp_content(result)

    assert is_error is True
    assert len(content) == 1
    assert isinstance(content[0], types.TextContent)
    payload = json.loads(content[0].text)
    assert "must contain bytes-like BEHAVIOR image data" in payload["error"]
