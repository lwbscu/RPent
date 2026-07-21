"""HTTP client for the BEHAVIOR Pi05 VLA server."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import httpx
import numpy as np

from robots.behavior.schemas import extract_policy_state, validate_action_chunk


def _png_b64(img: np.ndarray) -> str:
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class BehaviorVLAClient:
    """Client for a BEHAVIOR-compatible ``/predict`` endpoint."""

    def __init__(self, base_url: str, *, timeout_s: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        # The VLA endpoint is a runtime-owned local sidecar.  Ambient proxy
        # variables must never redirect it or make SOCKS extras a dependency.
        self._client = httpx.Client(timeout=timeout_s, trust_env=False)

    @property
    def endpoint(self) -> str:
        return self._base_url

    def healthz(self, *, timeout_ms: int | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if timeout_ms is not None:
            kwargs["timeout"] = timeout_ms / 1000.0
        resp = self._client.get(f"{self._base_url}/healthz", **kwargs)
        resp.raise_for_status()
        return resp.json()

    def wait_for_healthz(
        self,
        *,
        timeout_s: float = 600.0,
        poll_timeout_ms: int = 1000,
    ) -> None:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                self.healthz(timeout_ms=poll_timeout_ms)
                return
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        raise TimeoutError(
            f"BEHAVIOR vla server not healthy after {timeout_s:.0f}s "
            f"(last error: {last_err})"
        )

    def disable_actions(self, *, timeout_ms: int = 5000) -> dict[str, Any]:
        """Disable future policy inference while leaving ``healthz`` available."""

        resp = self._client.post(
            f"{self._base_url}/control/disable-actions",
            timeout=max(float(timeout_ms) / 1000.0, 0.001),
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("actions_enabled") is not False:
            raise RuntimeError(
                f"VLA server did not confirm action disable: {payload!r}"
            )
        return payload

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        main = np.asarray(env_obs["main_images"])
        wrists = np.asarray(env_obs["wrist_images"])
        if main.ndim != 3:
            raise ValueError(f"main_images must be [H,W,3], got {main.shape}")
        if wrists.ndim != 4 or wrists.shape[0] != 2:
            raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")

        states = np.asarray(env_obs["states"], dtype=np.float32)
        if states.ndim != 1:
            raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
        # Validate the raw proprio mapping without changing what crosses the
        # wire. RLinf's pi05_behavior transform performs the same extraction.
        extract_policy_state(states)

        body = {
            "instruction": str(env_obs.get("task_descriptions") or ""),
            "images": {
                "main": {"format": "png", "data": _png_b64(main)},
                "left_wrist": {"format": "png", "data": _png_b64(wrists[0])},
                "right_wrist": {"format": "png", "data": _png_b64(wrists[1])},
            },
            "state": [states.tolist()],
            "mode": mode,
        }
        resp = self._client.post(f"{self._base_url}/predict", json=body)
        if resp.status_code != 200:
            try:
                payload = resp.json()
                detail = payload.get("detail") or payload.get("error") or payload
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"BEHAVIOR VLA /predict failed (HTTP {resp.status_code}): {detail}"
            )
        payload = resp.json()
        action_batch = np.asarray(payload["actions"], dtype=np.float32)
        if action_batch.ndim != 3 or action_batch.shape[0] != 1:
            raise ValueError(
                "BEHAVIOR VLA response actions must be [1,T,23], "
                f"got {action_batch.shape}"
            )
        actions = validate_action_chunk(action_batch[0])
        return actions, {"shape": payload.get("shape"), "dtype": payload.get("dtype")}

    def close(self) -> None:
        self._client.close()
