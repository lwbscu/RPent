from __future__ import annotations

import threading

import pytest

from rpent.utils.socket_rpc import RpcError, SocketRpcClient, SocketRpcServer


class _Unserializable:
    def __reduce__(self):
        raise TypeError("deliberately not serializable")


def test_socket_rpc_returns_explicit_response_serialization_error():
    server = SocketRpcServer(
        ("127.0.0.1", 0),
        lambda method, args, kwargs: _Unserializable(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = SocketRpcClient(host, port)
        with pytest.raises(RpcError, match="response serialization failed") as exc:
            client.call("bad.result")
        assert "TypeError: deliberately not serializable" in str(exc.value)
        assert exc.value.server_traceback
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_socket_rpc_preserves_dispatch_errors():
    def dispatch(method, args, kwargs):
        raise ValueError("dispatch rejected")

    server = SocketRpcServer(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = SocketRpcClient(host, port)
        with pytest.raises(RpcError, match="dispatch rejected") as exc:
            client.call("rejected")
        assert "ValueError" in (exc.value.server_traceback or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
