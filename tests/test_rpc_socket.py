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


def test_socket_rpc_exact_concurrent_method_bypasses_serial_dispatch_lock():
    serial_started = threading.Event()
    release_serial = threading.Event()
    second_serial_done = threading.Event()
    calls = []

    def dispatch(method, args, kwargs):
        calls.append(("serial", method))
        if method == "serial.block":
            serial_started.set()
            if not release_serial.wait(timeout=2.0):
                raise TimeoutError("test serial dispatch was not released")
        return {"method": method}

    def concurrent_dispatch(method, args, kwargs):
        calls.append(("concurrent", method))
        return {"method": method, "concurrent": True}

    server = SocketRpcServer(
        ("127.0.0.1", 0),
        dispatch,
        concurrent_method="env.dashboard_prepare_manual_command",
        concurrent_dispatch=concurrent_dispatch,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address
    client = SocketRpcClient(host, port)

    blocked = threading.Thread(
        target=lambda: client.call("serial.block", timeout_s=3.0),
        daemon=True,
    )
    blocked.start()
    assert serial_started.wait(timeout=1.0)
    try:
        prepared = client.call(
            "env.dashboard_prepare_manual_command",
            kwargs={"background": True},
            timeout_s=1.0,
        )
        assert prepared["concurrent"] is True

        def second_serial():
            client.call("serial.second", timeout_s=3.0)
            second_serial_done.set()

        queued = threading.Thread(target=second_serial, daemon=True)
        queued.start()
        assert second_serial_done.wait(timeout=0.05) is False
        release_serial.set()
        blocked.join(timeout=1.0)
        queued.join(timeout=1.0)
        assert second_serial_done.is_set()
        assert calls[0] == ("serial", "serial.block")
        assert ("concurrent", "env.dashboard_prepare_manual_command") in calls
        assert ("serial", "serial.second") in calls
    finally:
        release_serial.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
