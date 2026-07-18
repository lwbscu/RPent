"""BEHAVIOR env RPC client."""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.rpc_driver.base import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 1800.0,
    "env.chunk_step": 1800.0,
}


class BehaviorEnvClient:
    """Remote implementation of the BEHAVIOR single-env protocol."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict[str, Any],
    ) -> None:
        self._client = client
        self.episode_done = False
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=_TIMEOUT_S["default"],
        )
        if server_meta != expected_meta:
            raise RuntimeError(
                f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}"
            )

    def reset(self) -> tuple[dict[str, Any], Any]:
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.episode_done = False
        return ret

    def chunk_step(self, actions) -> tuple[Any, Any, Any, Any, Any]:
        assert not self.episode_done, "env.chunk_step called after episode done"
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, info = ret
        if np.asarray(term).any() or np.asarray(trunc).any():
            self.episode_done = True
        if isinstance(info, dict) and bool((info.get("done") or {}).get("success")):
            self.episode_done = True
        return ret

    def get_env_meta(self) -> dict[str, Any]:
        return self._client.call("env.get_env_meta", timeout_s=_TIMEOUT_S["default"])

    def close(self) -> None:
        """Close only the client transport; runtime ownership stays with provider."""
        self._client.close()
