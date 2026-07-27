import httpx
import numpy as np
import pytest

from robots.behavior import vla_client


class _FailingHttpClient:
    def __init__(self) -> None:
        self.post_calls = 0

    def post(self, _url, *, json):
        del json
        self.post_calls += 1
        raise httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )

    def close(self) -> None:
        return None


def test_client_disables_keepalive_without_changing_timeout(monkeypatch):
    captured = {}

    class CapturingClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(vla_client.httpx, "Client", CapturingClient)

    client = vla_client.BehaviorVLAClient(
        "http://127.0.0.1:9123/",
        timeout_s=321.5,
    )
    client.close()

    assert captured["timeout"] == 321.5
    assert captured["trust_env"] is False
    limits = captured["limits"]
    assert limits.max_connections == 100
    assert limits.max_keepalive_connections == 0


def test_predict_remote_protocol_error_is_not_retried(monkeypatch):
    client = vla_client.BehaviorVLAClient("http://vla.example")
    client._client.close()
    failing_http = _FailingHttpClient()
    client._client = failing_http
    monkeypatch.setattr(vla_client, "_png_b64", lambda _image: "png")
    observation = {
        "main_images": np.zeros((3, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 3, 4, 3), dtype=np.uint8),
        "states": np.arange(256, dtype=np.float32),
        "task_descriptions": "pick up trash",
    }

    with pytest.raises(
        httpx.RemoteProtocolError,
        match="Server disconnected without sending a response",
    ):
        client.predict_action_batch(observation)

    assert failing_http.post_calls == 1
