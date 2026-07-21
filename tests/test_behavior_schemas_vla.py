import base64
import pickle
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import (
    BehaviorEnvFacade,
    _bootstrap_template_path,
    _configure_control_mode,
    _MainThreadDispatcher,
    _payload_intrinsics,
    _resize_video_tile,
    _sensor_camera_to_world,
    _sensor_intrinsics,
    _settle_visual_pipeline_after_restore,
    _wire_safe,
)
from robots.behavior.schemas import (
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    extract_policy_state,
    validate_action_chunk,
)
from robots.behavior.vla_client import BehaviorVLAClient


def test_planner_control_mode_uses_official_position_base_without_mutating_vla():
    def config():
        base = SimpleNamespace(
            name="HolonomicBaseJointController",
            motor_type="velocity",
            command_input_limits=[[-1.0] * 3, [1.0] * 3],
            command_output_limits=[[-0.75] * 3, [0.75] * 3],
            use_impedances=False,
        )
        robot = SimpleNamespace(
            type="R1Pro",
            controller_config=SimpleNamespace(base=base),
        )
        return SimpleNamespace(
            omni_config=SimpleNamespace(robots=[robot])
        )

    vla_cfg = config()
    _configure_control_mode(vla_cfg, "full_task_vla")
    assert vla_cfg.omni_config.robots[0].controller_config.base.motor_type == "velocity"

    planner_cfg = config()
    _configure_control_mode(planner_cfg, "planner_tools")
    base = planner_cfg.omni_config.robots[0].controller_config.base
    assert base.motor_type == "position"
    assert base.command_input_limits is None
    assert base.command_output_limits is None
    assert base.isaac_kp == 2_000_000.0
    assert base.isaac_kd == 100_000.0


def test_behavior_rpc_executes_env_method_on_dispatcher_thread():
    class _Env:
        called_on = None

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


@pytest.mark.parametrize(
    "method",
    ("env.dump_simulator_state", "env.restore_simulator_state", "env.start_video_segment"),
)
def test_acceptance_lifecycle_methods_are_not_public_rpc_tools(method):
    dispatcher = _MainThreadDispatcher(SimpleNamespace(), threading.Event())

    with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC method"):
        dispatcher._dispatch(method, (), {})


def test_restore_visual_pipeline_requires_three_render_updates_without_physics():
    class Simulator:
        def __init__(self):
            self.render_calls = 0

        def render(self):
            self.render_calls += 1

    simulator = Simulator()

    _settle_visual_pipeline_after_restore(simulator)

    assert simulator.render_calls == 3
    with pytest.raises(ValueError, match="at least 3 renders"):
        _settle_visual_pipeline_after_restore(simulator, render_iterations=2)


def test_behavior_env_client_sends_action_chunks_without_numpy_pickle_internals():
    expected_meta = {"activity_instance_id": 211, "seed": 211}

    class _RpcClient:
        def __init__(self):
            self.chunk_args = None

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            del kwargs, timeout_s
            if method == "env.get_env_meta":
                return expected_meta
            assert method == "env.chunk_step"
            self.chunk_args = args
            return {}, 0.0, False, False, {"done": {"success": False}}

    rpc = _RpcClient()
    client = BehaviorEnvClient(rpc, expected_meta=expected_meta)

    client.chunk_step(np.zeros((2, 23), dtype=np.float32))

    assert isinstance(rpc.chunk_args[0], list)
    assert len(rpc.chunk_args[0]) == 2
    assert all(isinstance(row, list) and len(row) == 23 for row in rpc.chunk_args[0])


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


def test_behavior_chunk_step_round_trips_official_success_with_simulator_object(
    monkeypatch,
):
    info = {
        "done": {"success": np.bool_(True)},
        "simulator_object": SimpleNamespace(callback=lambda: None),
    }

    class _DirectProcess:
        @staticmethod
        def step_env(_action, *, need_obs):
            assert need_obs is True
            return {}, np.array([1.0]), np.array([False]), np.array([False]), [info]

    wrapped_observation = {
        "main_images": np.zeros((1, 3, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 3, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 256), dtype=np.float32),
        "task_descriptions": ["turn on the radio"],
    }
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._done = False
    facade._env_steps = 0
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

    result = facade.chunk_step(np.zeros((1, 23), dtype=np.float32))
    round_tripped = pickle.loads(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))

    assert round_tripped[4]["done"]["success"] is True
    assert round_tripped[4]["simulator_object"].startswith("<unserializable:")
    assert round_tripped[4]["_rpent"] == {"executed_steps": 1}
    assert facade._done is True


def test_behavior_chunk_renders_every_four_steps_and_on_last_step(monkeypatch):
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
    facade._done = False
    facade._env_steps = 0
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

    result = facade.chunk_step(np.zeros((5, 23), dtype=np.float32))

    assert need_obs_calls == [False, False, False, True, True]
    assert result[4]["_rpent"] == {"executed_steps": 5}


def test_behavior_planner_step_samples_rgbd_every_four_steps(monkeypatch):
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

    for _ in range(4):
        facade.planner_step(np.zeros((1, 23), dtype=np.float32))

    assert need_obs_calls == [False] * 3 + [True]


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
    facade._env_steps = 20
    facade._control_mode = "planner_tools"
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
    facade._env_steps = 20
    facade._control_mode = "planner_tools"
    facade._frame_cache = FrameCache(ttl_s=100.0)
    facade._sensor_for_camera = lambda _camera: None
    facade._env = _BehaviorEnv()
    facade._planner = _Planner(facade._frame_cache)
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

    assert facade._env.omnigibson_env.calls == 1
    assert refreshed_head["frame_id"] != old_frame_id
    assert refreshed_head["capture_group"]["id"] == refreshed_wrist["capture_group"]["id"]
    assert refreshed_head["capture_group"]["cameras"] == refreshed_wrist["capture_group"]["cameras"]
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


def test_action_validation_rejects_nan():
    actions = np.zeros((2, 23), dtype=np.float32)
    actions[1, 7] = np.nan

    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_action_chunk(actions)


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
    assert set(body) == {"instruction", "images", "state", "mode"}
    assert body["instruction"] == "turn on the radio"
    assert body["mode"] == "eval"
    assert body["state"] == [raw.tolist()]
    assert list(body["images"]) == ["main", "left_wrist", "right_wrist"]
    for image in body["images"].values():
        assert image["format"] == "png"
        assert base64.b64decode(image["data"]).startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(predicted, actions[0])
    assert metadata == {"shape": [1, 2, 23], "dtype": "float32"}
