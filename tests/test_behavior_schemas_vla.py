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
    _MainThreadDispatcher,
    _payload_intrinsics,
    _sensor_intrinsics,
    _wire_safe,
)
from robots.behavior.schemas import (
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    extract_policy_state,
    validate_action_chunk,
)
from robots.behavior.vla_client import BehaviorVLAClient


def test_behavior_rpc_executes_env_method_on_dispatcher_thread():
    class _Env:
        called_on = None

        def ping(self):
            self.called_on = threading.get_ident()
            return "pong"

    env = _Env()
    shutdown_event = threading.Event()
    dispatcher = _MainThreadDispatcher(env, shutdown_event)
    result = {}
    submitter = threading.Thread(
        target=lambda: result.setdefault("value", dispatcher.submit("env.ping", (), {}))
    )
    submitter.start()

    assert dispatcher.process_next(timeout_s=1.0)
    submitter.join(timeout=1.0)

    assert not submitter.is_alive()
    assert result == {"value": "pong"}
    assert env.called_on == threading.get_ident()


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
