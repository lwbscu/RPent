# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
import base64
import hashlib
import pickle
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.env_server import (
    BehaviorEnvFacade,
    _bootstrap_template_path,
    _MainThreadDispatcher,
    _payload_intrinsics,
    _raw_success,
    _resize_video_tile,
    _resolve_env_task_identity,
    _sensor_camera_to_world,
    _sensor_intrinsics,
    _wire_safe,
)
from robots.behavior.schemas import (
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    behavior_tool_specs_for_task,
    extract_policy_state,
    validate_action_chunk,
)
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
)
from robots.behavior.vla_client import BehaviorVLAClient


def _initialize_facade_runtime_fields(facade, *, max_episode_steps=100):
    facade._meta = {"max_episode_steps": int(max_episode_steps)}
    facade._motion_in_flight = False
    facade._official_success_latched = False
    facade._official_success_receipt = None
    facade._official_success_receipt_path = None
    facade._last_info = {"done": {"success": False}}
    facade._controller_state = "planner"
    facade._attempt_index = 1
    facade._attempt_nonce = "attempt"
    facade._run_nonce = "run"
    facade._pending_vla_visual_authorization = None
    facade._gripper_latch = {"left": 1.0, "right": 1.0}
    facade._public_observed_frame_ids = set()
    facade._projection_receipts = {}
    facade._consumed_projection_receipts = set()


def test_pi0_visual_review_stays_inside_current_attempt_root(tmp_path):
    capture = {"id": "capture:32:test"}
    planner = SimpleNamespace(
        observe=lambda camera: {
            "_image_bytes": f"{camera}-png".encode(),
            "capture_group": capture,
            "frame_id": f"{camera}:32:test",
        }
    )
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._output_dir = tmp_path / "attempt_001"
    facade._active_vla_call_index = 1
    facade._env_steps = 32
    facade._planner = planner

    result = facade._persist_pi0_nav_pick_views(
        chunk_index=1,
        validator={"local_grasp_success": False},
    )

    expected = facade._output_dir / "vla_calls/call_001/visual_review/chunk_0001"
    assert result["metadata_path"] == str(expected / "metadata.json")
    assert set(result["views"]) == {"head", "left_wrist", "right_wrist"}
    assert result["views"]["right_wrist"]["path"] == str(expected / "right_wrist.png")
    assert {path.name for path in expected.iterdir()} == {
        "head.png",
        "left_wrist.png",
        "right_wrist.png",
        "metadata.json",
    }
    assert not (facade._output_dir / "attempts").exists()


def test_behavior_rpc_executes_env_method_on_dispatcher_thread():
    class _Env:
        called_on = None

        @staticmethod
        def _assert_rpc_lifecycle(_method):
            return None

        def get_env_meta(self):
            self.called_on = threading.get_ident()
            return "pong"

    env = _Env()
    shutdown_event = threading.Event()
    dispatcher = _MainThreadDispatcher(env, shutdown_event)
    result = {}
    submitter = threading.Thread(
        target=lambda: result.setdefault(
            "value", dispatcher.submit("env.get_env_meta", (), {})
        )
    )
    submitter.start()

    assert dispatcher.process_next(timeout_s=1.0)
    submitter.join(timeout=1.0)

    assert not submitter.is_alive()
    assert result == {"value": "pong"}
    assert env.called_on == threading.get_ident()


def test_behavior_wire_info_replaces_only_unpickleable_leaves():
    payload = {
        "done": {"success": np.bool_(False)},
        "scores": np.array([1.0, 2.0], dtype=np.float32),
        "simulator_object": lambda: None,
        "object_array": np.array([np.int64(3), lambda: None], dtype=object),
    }

    safe = _wire_safe(payload)

    assert safe["done"]["success"] is False
    np.testing.assert_array_equal(safe["scores"], payload["scores"])
    assert safe["simulator_object"].startswith("<unserializable:")
    assert safe["object_array"][0] == 3
    assert safe["object_array"][1].startswith("<unserializable:")
    pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)


def test_behavior_planner_step_does_not_capture_between_waypoints(monkeypatch):
    need_obs_calls = []

    class _DirectProcess:
        @staticmethod
        def step_env(_action, *, need_obs):
            need_obs_calls.append(need_obs)
            raw = {} if need_obs else None
            info = {"done": {"success": False}}
            return raw, np.array([0.0]), np.array([False]), np.array([False]), [info]

    wrapped_observation = {
        "main_images": np.zeros((1, 3, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 3, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["turn on the radio"],
    }
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    _initialize_facade_runtime_fields(facade)
    facade._done = False
    facade._env_steps = 0
    facade._planner_video_interval_steps = 4
    facade._last_observation = wrapped_observation
    facade._env = SimpleNamespace(
        _direct_process=_DirectProcess(),
        _wrap_obs=lambda _raw: wrapped_observation,
    )
    facade._record_rgbd_frames = lambda _raw, _wrapped: None
    facade._append_video = lambda _observation: None
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32=np.float32,
            as_tensor=lambda value, dtype: np.asarray(value, dtype=dtype),
            is_tensor=lambda _value: False,
        ),
    )

    action = np.zeros((1, 23), dtype=np.float32)
    action[:, ENV_ACTION_SEGMENTS["left_gripper"]] = -0.5
    action[:, ENV_ACTION_SEGMENTS["right_gripper"]] = 0.25
    for _ in range(4):
        facade.planner_step(action)

    assert need_obs_calls == [False] * 4
    assert facade._gripper_latch == {"left": -0.5, "right": 0.25}


def test_planner_video_resizes_wrist_tile_without_changing_rgb_contract():
    wrist = np.zeros((2, 3, 4), dtype=np.uint8)
    wrist[..., 0] = 255

    tile = _resize_video_tile(wrist, height=6, width=8)

    assert tile.shape == (6, 8, 3)
    assert tile.dtype == np.uint8
    assert np.all(tile[..., 0] == 255)
    assert np.all(tile[..., 1:] == 0)


def test_invalid_payload_intrinsics_fall_through_to_valid_sensor_intrinsics():
    bad_payload = {"intrinsics": np.diag([0.0, 0.0, 1.0])}

    class _Sensor:
        intrinsic_matrix = np.array(
            [[300.0, 0.0, 160.0], [0.0, 301.0, 120.0], [0.0, 0.0, 1.0]]
        )

    payload_intrinsics = _payload_intrinsics(bad_payload, rgb_shape=(240, 320, 3))
    sensor_intrinsics = _sensor_intrinsics(_Sensor(), rgb_shape=(240, 320, 3))

    assert payload_intrinsics is None
    assert sensor_intrinsics is not None
    assert sensor_intrinsics.fx == 300.0
    assert sensor_intrinsics.fy == 301.0


def test_sensor_intrinsics_fall_back_to_verified_usd_camera_physical_attributes():
    class _Sensor:
        intrinsic_matrix = np.diag([0.0, 0.0, 1.0])
        focal_length = 20.0
        horizontal_aperture = 40.0

        @staticmethod
        def get_attribute(name):
            return {
                "verticalAperture": 30.0,
                "horizontalApertureOffset": 0.0,
                "verticalApertureOffset": 0.0,
            }[name]

    intrinsics = _sensor_intrinsics(_Sensor(), rgb_shape=(300, 400, 3))

    assert intrinsics is not None
    assert intrinsics.fx == 200.0
    assert intrinsics.fy == 200.0
    assert intrinsics.cx == 200.0
    assert intrinsics.cy == 150.0


def test_sensor_camera_pose_prefers_render_synchronous_kit_view_transform():
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]
    kit_view = np.linalg.inv(camera_to_world).T.reshape(-1)

    class _Sensor:
        camera_parameters = {"cameraViewTransform": kit_view}

        @staticmethod
        def get_position_orientation():
            return np.array([9.0, 9.0, 9.0]), np.array([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(_sensor_camera_to_world(_Sensor()), camera_to_world)


def test_sensor_camera_pose_falls_back_when_first_kit_view_is_zero():
    class _Sensor:
        camera_parameters = {"cameraViewTransform": np.zeros(16)}

        @staticmethod
        def get_position_orientation():
            return np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0])

    expected = np.eye(4, dtype=np.float64)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(_sensor_camera_to_world(_Sensor()), expected)


def _same_step_raw_rgbd(*, include_right=True, privileged_pose=None):
    intrinsics = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    sensors = {}
    names = {
        "head": "robot_r1:zed_link:Camera:0",
        "left_wrist": "robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    }
    for index, (camera, name) in enumerate(names.items()):
        if camera == "right_wrist" and not include_right:
            continue
        sensors[name] = {
            "rgb": np.full((5, 5, 3), index, dtype=np.uint8),
            "depth_linear": np.full((5, 5), index + 1.0, dtype=np.float32),
            "intrinsics": intrinsics,
            "camera_to_world": np.eye(4, dtype=np.float64),
            # A hostile wrapper may attach privileged fields to raw observations;
            # the planner capture boundary must never forward them.
            "object_pose": privileged_pose,
        }
    return [{"robot_r1": sensors}]


def test_env_records_atomic_three_camera_group_with_compact_proprio_only():
    from robots.behavior.camera_geometry import FrameCache

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 12
    facade._frame_cache = FrameCache(ttl_s=100.0)
    facade._sensor_for_camera = lambda _camera: None
    raw_proprio = np.arange(256, dtype=np.float32)

    facade._record_rgbd_frames(
        _same_step_raw_rgbd(privileged_pose=[9.0, 8.0, 7.0]),
        {"states": raw_proprio},
    )

    frames = {
        camera: facade._frame_cache.latest(camera)
        for camera in ("head", "left_wrist", "right_wrist")
    }
    assert len({frame.capture_group_id for frame in frames.values()}) == 1
    assert {frame.step_index for frame in frames.values()} == {12}
    assert len({frame.timestamp_s for frame in frames.values()}) == 1
    payload = facade._frame_cache.observe_payload("head")
    assert payload["capture_group"]["sim_step"] == 12
    assert payload["proprio"] == {
        "values": extract_policy_state(raw_proprio).astype(float).tolist(),
        "dimension": 23,
        "layout": "POLICY_STATE_SEGMENTS",
        "segments": {
            "base": [0, 3],
            "trunk": [3, 7],
            "left_arm": [7, 14],
            "right_arm": [14, 21],
            "left_gripper": [21, 22],
            "right_gripper": [22, 23],
        },
    }
    assert "object_pose" not in payload
    assert "camera_to_world" not in payload
    assert "depth_m" not in payload


def test_env_missing_camera_does_not_partially_replace_previous_capture_group():
    from robots.behavior.camera_geometry import FrameCache

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    _initialize_facade_runtime_fields(facade)
    facade._env_steps = 20
    facade._frame_cache = FrameCache(ttl_s=100.0)
    facade._sensor_for_camera = lambda _camera: None
    observation = {"states": np.arange(256, dtype=np.float32)}
    facade._record_rgbd_frames(_same_step_raw_rgbd(), observation)
    first_group = facade._frame_cache.latest("head").capture_group_id

    facade._env_steps = 21
    facade._record_rgbd_frames(
        _same_step_raw_rgbd(include_right=False),
        observation,
    )

    assert {
        facade._frame_cache.latest(camera).capture_group_id
        for camera in ("head", "left_wrist", "right_wrist")
    } == {first_group}
    assert {
        facade._frame_cache.latest(camera).step_index
        for camera in ("head", "left_wrist", "right_wrist")
    } == {20}


def test_env_observe_refreshes_aged_rgbd_before_ttl_without_stepping_simulator():
    from robots.behavior.camera_geometry import CameraGeometryError, FrameCache

    raw_observation = _same_step_raw_rgbd()[0]

    class _OmniEnv:
        calls = 0

        def get_obs(self):
            self.calls += 1
            return raw_observation, {"sensor": "current"}

    class _BehaviorEnv:
        omnigibson_env = _OmniEnv()

        @staticmethod
        def _wrap_obs(observations):
            assert observations == [raw_observation]
            return {
                "main_images": np.zeros((1, 5, 5, 3), dtype=np.uint8),
                "wrist_images": np.zeros((1, 2, 5, 5, 3), dtype=np.uint8),
                "states": np.arange(256, dtype=np.float32)[None, :],
                "task_descriptions": ["turn on radio"],
            }

    class _Planner:
        def __init__(self, cache):
            self.cache = cache

        def observe(self, camera):
            return self.cache.observe_payload(camera)

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    _initialize_facade_runtime_fields(facade)
    facade._env_steps = 20
    facade._frame_cache = FrameCache(ttl_s=100.0)
    facade._sensor_for_camera = lambda _camera: None
    facade._env = _BehaviorEnv()
    facade._planner = _Planner(facade._frame_cache)
    facade._attachment_runtime_facts = lambda: {
        "held_count": 1,
        "hands": ["left"],
        "ambiguous": False,
    }
    facade._physical_gripper_opening = lambda hand: 0.0 if hand == "left" else 1.0
    render_only_calls = []
    facade._render_only_for_hand_geometry = lambda: render_only_calls.append(True)
    facade._last_observation = None
    facade._record_rgbd_frames(
        [raw_observation],
        {"states": np.arange(256, dtype=np.float32)},
    )
    old_frame = facade._frame_cache.latest("head")
    old_frame_id = old_frame.frame_id
    for camera in ("head", "left_wrist", "right_wrist"):
        facade._frame_cache.latest(camera).timestamp_s -= 6.0

    refreshed_head = facade.observe("head")
    refreshed_wrist = facade.observe("left_wrist")
    assert len(render_only_calls) == 4

    # Head refreshes the aged capture with one render-only synchronization.
    # The wrist then takes a second capture with the three stronger
    # hand-geometry render synchronizations.
    assert facade._env.omnigibson_env.calls == 2
    assert refreshed_head["frame_id"] != old_frame_id
    assert (
        refreshed_head["capture_group"]["id"] != refreshed_wrist["capture_group"]["id"]
    )
    assert "cameras" not in refreshed_head["capture_group"]
    assert "cameras" not in refreshed_wrist["capture_group"]
    assert refreshed_head["capture_group"]["sim_step"] == 20
    with pytest.raises(CameraGeometryError, match="stale frame_id"):
        facade._frame_cache.get_current("head", old_frame_id)


def test_bootstrap_template_uses_instance_zero_without_changing_target(tmp_path):
    instance_dir = tmp_path / "radio_instances"
    instance_dir.mkdir()
    template = tmp_path / "house_task_turning_on_radio_0_0_template.json"
    template.write_text("{}", encoding="utf-8")

    resolved = _bootstrap_template_path(
        instance_dir,
        scene_model="house",
        task_name="turning_on_radio",
        activity_definition_id=0,
    )

    assert resolved == template


def test_bootstrap_template_requires_a_full_instance_zero_template(tmp_path):
    instance_dir = tmp_path / "radio_instances"
    instance_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="bootstrap scene template"):
        _bootstrap_template_path(
            instance_dir,
            scene_model="house",
            task_name="turning_on_radio",
            activity_definition_id=0,
        )


def _task_meta(
    tmp_path,
    *,
    task_name="picking_up_trash",
    task=1,
    activity_definition_id=0,
    activity_instance_id=196,
    scene_model="house_double_floor_lower",
    public_seed=0,
):
    spec = (
        PICKING_UP_TRASH_TASK_SPEC
        if task_name == "picking_up_trash"
        else TURNING_ON_RADIO_TASK_SPEC
    )
    return {
        "task_name": task_name,
        "task": task,
        "activity_definition_id": activity_definition_id,
        "activity_instance_id": activity_instance_id,
        "activity_instance_dir": str(tmp_path / spec.state_dir_name),
        "scene_model": scene_model,
        "public_seed": public_seed,
    }


def test_env_identity_validates_task_scoped_public_mapping_before_construction(
    tmp_path,
):
    spec, identity = _resolve_env_task_identity(_task_meta(tmp_path))

    assert spec is PICKING_UP_TRASH_TASK_SPEC
    assert identity == ("picking_up_trash", 0, 196)

    with pytest.raises(ValueError, match="public s0, not s1"):
        _resolve_env_task_identity(_task_meta(tmp_path, public_seed=1))
    with pytest.raises(ValueError, match="requires activity_definition_id 0"):
        _resolve_env_task_identity(_task_meta(tmp_path, activity_definition_id=1))
    with pytest.raises(ValueError, match="requires scene_model"):
        _resolve_env_task_identity(_task_meta(tmp_path, scene_model="wrong_scene"))


def test_task_identity_classifies_same_native_instance_per_selected_task(tmp_path):
    trash, trash_identity = _resolve_env_task_identity(
        _task_meta(tmp_path, activity_instance_id=242)
    )
    radio, radio_identity = _resolve_env_task_identity(
        _task_meta(
            tmp_path,
            task_name="turning_on_radio",
            task=0,
            activity_instance_id=242,
        )
    )

    assert trash.classify_instance(242).kind == "candidate"
    assert radio.classify_instance(242).kind == "explore"
    assert trash_identity == ("picking_up_trash", 0, 242)
    assert radio_identity == ("turning_on_radio", 0, 242)


def test_trash_schema_and_runtime_reject_radio_only_policies_before_side_effects():
    specs = behavior_tool_specs_for_task(PICKING_UP_TRASH_TASK_SPEC)
    encoded = repr(specs).lower()
    for forbidden in (
        "radio_tipped_flat",
        "visual_radio_tipped_flat",
        "opposite_surface_confirmed",
        "target_bearing_surface_confirmed",
    ):
        assert forbidden not in encoded
    assert (
        "terminal_failure"
        not in specs["save_robot_state_checkpoint"]["input_schema"]["properties"]
    )
    assert "frame_review" not in specs["observe"]["input_schema"]["properties"]

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    with pytest.raises(
        ValueError,
        match="picking_up_trash does not define a visual terminal-failure policy",
    ):
        facade.save_robot_state_checkpoint(
            terminal_failure={
                "condition": "radio_tipped_flat",
                "cause": "dropped_out_of_gripper",
                "camera": "head",
                "frame_id": "head:evidence",
            }
        )
    with pytest.raises(ValueError, match="does not define frame review"):
        facade._review_public_observation(
            requested_camera="head",
            frame_review={
                "frame_id": "head:evidence",
                "assessment": "opposite_surface_confirmed",
            },
        )


def test_trash_attached_hand_rotate_does_not_create_surface_regression_receipt():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._task_spec = PICKING_UP_TRASH_TASK_SPEC
    facade._env_steps = 7
    facade._official_success_latched = False
    facade._last_info = {"done": {"success": False}}
    facade._completed_opposite_surface_cycles = [{"cycle_id": "radio-only"}]
    facade._authorize_analytic_hand = lambda *_args: (
        "right",
        "llm_visual_hand_selection",
        {"kind": "visual_hand_authorization"},
    )
    facade._revalidate_analytic_selection = lambda **_kwargs: (
        "right",
        "llm_visual_hand_selection",
        {"kind": "visual_hand_authorization"},
    )
    attached = object()
    facade._attachment_runtime_facts = lambda: {
        "available": True,
        "attachment_count": 1,
        "identity_conflict": False,
        "hands": ["right"],
        "attached_objects": {"right": attached},
        "by_hand": {
            "left": {"attached": False},
            "right": {"attached": True},
        },
    }
    facade._attachment_lineage_fingerprint = lambda *_args, **_kwargs: "attachment"
    facade._attachment_fingerprint_snapshot = lambda _facts=None: {
        "available": True,
        "hands": ["right"],
        "env_step": 7,
        "fingerprints": {"left": None, "right": "attachment"},
    }
    facade._switch_controller = lambda *_args, **_kwargs: None
    facade._require_planner = lambda: SimpleNamespace(
        rotate_wrist=lambda **_kwargs: {
            "primitive_success": True,
            "metrics": {},
        }
    )
    facade._analytic_public_result = lambda result, **_kwargs: dict(result)

    result = facade.rotate_wrist(
        hand="right",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.1],
        visual_hand_check={
            "camera": "head",
            "frame_id": "head:7:fresh",
            "selected_hand": "right",
            "assessment": "selected_hand_visually_confirmed",
        },
    )

    assert result["primitive_success"] is True
    assert "attached_rotate_receipt" not in result
    assert facade._completed_opposite_surface_cycles == [{"cycle_id": "radio-only"}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (np.bool_(True), True),
        (False, False),
        (np.bool_(False), False),
        (1, False),
        ("false", False),
        ([False], False),
    ],
)
def test_raw_success_requires_a_boolean_true(value, expected):
    assert _raw_success({"done": {"success": value}}) is expected


def test_policy_state_and_env_action_have_exact_distinct_23d_orders():
    assert list(POLICY_STATE_SEGMENTS.items()) == [
        ("base", slice(0, 3)),
        ("trunk", slice(3, 7)),
        ("left_arm", slice(7, 14)),
        ("right_arm", slice(14, 21)),
        ("left_gripper", slice(21, 22)),
        ("right_gripper", slice(22, 23)),
    ]
    assert list(ENV_ACTION_SEGMENTS.items()) == [
        ("base", slice(0, 3)),
        ("trunk", slice(3, 7)),
        ("left_arm", slice(7, 14)),
        ("left_gripper", slice(14, 15)),
        ("right_arm", slice(15, 22)),
        ("right_gripper", slice(22, 23)),
    ]
    assert POLICY_STATE_SEGMENTS != ENV_ACTION_SEGMENTS


def test_extract_policy_state_uses_raw_r1pro_indices_in_policy_order():
    raw = np.arange(256, dtype=np.float32)

    compact = extract_policy_state(raw)

    expected = np.concatenate(
        [
            raw[253:256],
            raw[236:240],
            raw[158:165],
            raw[197:204],
            [raw[193:195].sum()],
            [raw[232:234].sum()],
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(compact, expected)
    assert compact.shape == (23,)


@pytest.mark.parametrize(
    "bad_actions",
    [
        np.zeros(23, dtype=np.float32),
        np.zeros((0, 23), dtype=np.float32),
        np.zeros((1, 22), dtype=np.float32),
        np.zeros((1, 24), dtype=np.float32),
    ],
)
def test_action_validation_rejects_non_t_by_23_shapes(bad_actions):
    with pytest.raises(ValueError, match=r"must be \[T,23\]"):
        validate_action_chunk(bad_actions)


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, url, *, json):
        self.requests.append((url, json))
        return _Response(self.payload)


def test_behavior_vla_wire_keeps_three_cameras_raw_proprio_and_batched_actions():
    raw = np.arange(256, dtype=np.float32)
    actions = np.arange(2 * 23, dtype=np.float32).reshape(1, 2, 23)
    http = _HttpClient(
        {"actions": actions.tolist(), "shape": [1, 2, 23], "dtype": "float32"}
    )
    client = BehaviorVLAClient("http://vla.example/")
    client._client.close()
    client._client = http
    observation = {
        "main_images": np.zeros((3, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 3, 4, 3), dtype=np.uint8),
        "states": raw,
        "task_descriptions": "turn on the radio",
    }

    predicted, metadata = client.predict_action_batch(observation)

    assert len(http.requests) == 1
    url, body = http.requests[0]
    assert url == "http://vla.example/predict"
    assert set(body) == {"instruction", "images", "state", "mode", "binding_id"}
    assert body["binding_id"] is None
    assert body["instruction"] == "turn on the radio"
    assert body["mode"] == "eval"
    assert body["state"] == [raw.tolist()]
    assert list(body["images"]) == ["main", "left_wrist", "right_wrist"]
    for image in body["images"].values():
        assert image["format"] == "png"
        assert base64.b64decode(image["data"]).startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(predicted, actions[0])
    assert metadata == {"shape": [1, 2, 23], "dtype": "float32"}


def test_vla_attempt_binding_rejects_stale_identity(monkeypatch):
    import robots.behavior.vla_server as server

    monkeypatch.setattr(server, "_ACTION_BINDING_ID", "job.a2")
    server._require_matching_binding("job.a2")
    with pytest.raises(ValueError, match="binding mismatch"):
        server._require_matching_binding("job.a1")
    with pytest.raises(ValueError, match="binding mismatch"):
        server._require_matching_binding(None)


def test_vla_client_sends_binding_on_bidirectional_gate_calls():
    class ControlHttp:
        def __init__(self):
            self.requests = []

        def post(self, url, *, json=None, timeout=None):
            self.requests.append((url, json, timeout))
            if url.endswith("bind-actions"):
                digest = hashlib.sha256(json["binding_id"].encode()).hexdigest()
                return _Response(
                    {
                        "actions_enabled": False,
                        "binding_digest": digest,
                    }
                )
            return _Response({"actions_enabled": url.endswith("enable-actions")})

    http = ControlHttp()
    client = BehaviorVLAClient("http://vla.example", binding_id="job.a1")
    client._client.close()
    client._client = http

    client.enable_actions()
    client.disable_actions()
    client.bind_actions("job.a2")
    client.enable_actions()

    assert http.requests[0][1] == {"binding_id": "job.a1"}
    assert http.requests[1][1] == {"binding_id": "job.a1"}
    assert http.requests[2][1] == {"binding_id": "job.a2"}
    assert http.requests[3][1] == {"binding_id": "job.a2"}
