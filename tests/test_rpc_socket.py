import threading

import pytest

from rpent.rpc_driver.socket import RpcError, SocketRpcClient, SocketRpcServer


def test_rpc_reports_response_serialization_failure_instead_of_closing_socket():
    server = SocketRpcServer(
        ("127.0.0.1", 0), lambda _method, _args, _kwargs: lambda: None
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SocketRpcClient(*server.server_address)
        with pytest.raises(RpcError, match="response serialization failed"):
            client.call("unpickleable")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
