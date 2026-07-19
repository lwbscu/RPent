import base64
import pickle
import threading

import numpy as np
import pytest

from robots.behavior.env_server import (
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
        target=lambda: result.setdefault(
            "value", dispatcher.submit("env.ping", (), {})
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
