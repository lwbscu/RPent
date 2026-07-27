"""Pi0.5 server for BEHAVIOR; this process never imports OmniGibson."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from robots.behavior.policy_checkpoint import (
    SHARED_POLICY_CHECKPOINT_PATH,
    validate_policy_checkpoint,
)
from robots.behavior.schemas import ACTION_DIM, DEFAULT_ACTION_CHUNK
from rpent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger

logger = get_logger("behavior_vla_server")
RPENT_ROOT = get_repo_root()
RLINF_ROOT = get_rlinf_repo_path() or (RPENT_ROOT.parent / "RLinf_agentic_push")
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

NORM_STATS_REL = Path("assets/behavior-1k/2025-challenge-demos/norm_stats.json")


class ImageBlock(BaseModel):
    format: str = "png"
    data: str


class PredictRequest(BaseModel):
    instruction: str
    images: dict[str, ImageBlock]
    state: list[list[float]]
    mode: str = "eval"
    binding_id: str | None = None


class BindingRequest(BaseModel):
    binding_id: str


_MODEL: Any = None
_MODEL_META: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_ACTIONS_ENABLED = True
_ACTIONS_LOCK = threading.Lock()
_ACTION_BINDING_ID: str | None = None


def _binding_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_matching_binding(value: str | None) -> None:
    if _ACTION_BINDING_ID is None:
        if value is not None:
            raise ValueError("VLA server is not bound to this attempt")
        return
    if value != _ACTION_BINDING_ID:
        raise ValueError("VLA attempt binding mismatch")


def validate_checkpoint(path: str | Path) -> Path:
    """Return the verified shared checkpoint root.

    Kept as a path-returning compatibility shim for callers that build the
    model configuration directly.  The authoritative validator also verifies
    the checkpoint fingerprint.
    """

    return Path(validate_policy_checkpoint(path).resolved_path)


def build_model_config(checkpoint: str | Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": None,
            "openpi_data": {
                "extra_delta_transform": False,
                "extract_state_from_proprio": True,
                "use_all_wrist_images": True,
                "use_quantile_norm": True,
            },
            "openpi": {
                "config_name": "pi05_behavior",
                "num_images_in_input": 3,
                "action_dim": 32,
                "action_horizon": DEFAULT_ACTION_CHUNK,
                "action_chunk": DEFAULT_ACTION_CHUNK,
                "action_env_dim": ACTION_DIM,
                "num_steps": 4,
                "add_value_head": False,
                "noise_level": 0.0,
                "noise_method": "flow_sde",
                "joint_logprob": False,
            },
        }
    )


def load_model(checkpoint: str | Path, *, seed: int) -> None:
    global _ACTION_BINDING_ID, _ACTIONS_ENABLED, _MODEL, _MODEL_META
    import torch
    from rlinf.models.embodiment.openpi import get_model

    checkpoint_binding = validate_policy_checkpoint(checkpoint)
    checkpoint = Path(checkpoint_binding.resolved_path)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.time()
    model = get_model(build_model_config(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = model.to(device).eval()
    with _ACTIONS_LOCK:
        _ACTIONS_ENABLED = True
        _ACTION_BINDING_ID = None
    _MODEL_META = {
        "status": "ok",
        "config_name": "pi05_behavior",
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "action_dim": ACTION_DIM,
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_binding": checkpoint_binding.as_dict(),
        "seed": int(seed),
        "load_elapsed_s": round(time.time() - started, 2),
    }
    logger.info(
        "Pi0.5 ready: config=pi05_behavior checkpoint=%s device=%s elapsed=%.1fs",
        checkpoint,
        device,
        _MODEL_META["load_elapsed_s"],
    )


def _decode_image(block: dict[str, Any]) -> np.ndarray:
    import imageio.v2 as imageio

    if str(block.get("format", "png")).lower() != "png":
        raise ValueError("only PNG image blocks are supported")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block is missing base64 data")
    image = np.asarray(imageio.imread(io.BytesIO(base64.b64decode(data))))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be [H,W,3], got {image.shape}")
    return image.astype(np.uint8, copy=False)


def build_env_observation(request: dict[str, Any]) -> dict[str, Any]:
    import torch

    images = request.get("images") or {}
    required = ("main", "left_wrist", "right_wrist")
    missing = [name for name in required if name not in images]
    if missing:
        raise ValueError(f"missing image(s): {missing}")
    state = np.asarray(request.get("state"), dtype=np.float32)
    if state.ndim != 2 or state.shape[0] != 1 or state.shape[1] < 256:
        raise ValueError(
            "state must contain one raw R1Pro proprio vector [1,N>=256], "
            f"got {state.shape}"
        )
    main = _decode_image(images["main"])
    left = _decode_image(images["left_wrist"])
    right = _decode_image(images["right_wrist"])
    return {
        "main_images": torch.from_numpy(main[None]),
        "wrist_images": torch.from_numpy(np.stack([left, right], axis=0)[None]),
        "states": torch.from_numpy(state),
        "task_descriptions": [str(request.get("instruction") or "")],
        "extra_view_images": None,
    }


def build_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="RPent BEHAVIOR Pi0.5")

    @app.get("/healthz")
    def healthz():
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        with _ACTIONS_LOCK:
            actions_enabled = bool(_ACTIONS_ENABLED)
            binding_digest = _binding_digest(_ACTION_BINDING_ID)
        return {
            **_MODEL_META,
            "pid": os.getpid(),
            "actions_enabled": actions_enabled,
            "binding_digest": binding_digest,
        }

    @app.post("/control/disable-actions")
    def disable_actions(request: BindingRequest | None = None):
        """Idempotently gate inference after a controller handoff."""

        global _ACTIONS_ENABLED
        # Lock ordering matches predict(): once this returns, no inference is
        # in flight and no later request can enter the model.
        with _MODEL_LOCK, _ACTIONS_LOCK:
            if request is not None:
                try:
                    _require_matching_binding(request.binding_id)
                except ValueError as error:
                    raise HTTPException(status_code=409, detail=str(error)) from error
            _ACTIONS_ENABLED = False
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": False,
            "binding_digest": _binding_digest(_ACTION_BINDING_ID),
        }

    @app.post("/control/bind-actions")
    def bind_actions(request: BindingRequest):
        """Replace the attempt binding only while inference is disabled."""

        global _ACTION_BINDING_ID
        binding_id = request.binding_id.strip()
        if not binding_id or len(binding_id) > 256:
            raise HTTPException(status_code=400, detail="invalid binding_id")
        with _MODEL_LOCK, _ACTIONS_LOCK:
            if _ACTIONS_ENABLED:
                raise HTTPException(
                    status_code=409,
                    detail="disable VLA actions before binding a fresh attempt",
                )
            _ACTION_BINDING_ID = binding_id
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": False,
            "binding_digest": _binding_digest(binding_id),
        }

    @app.post("/control/enable-actions")
    def enable_actions(request: BindingRequest | None = None):
        """Idempotently re-arm inference after the env confirms it is safe."""

        global _ACTIONS_ENABLED
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        # Use the same lock order as predict() / disable_actions(). Once this
        # returns there was no in-flight inference during the gate transition.
        with _MODEL_LOCK, _ACTIONS_LOCK:
            try:
                _require_matching_binding(
                    request.binding_id if request is not None else None
                )
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            _ACTIONS_ENABLED = True
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": True,
            "binding_digest": _binding_digest(_ACTION_BINDING_ID),
        }

    @app.post("/predict")
    def predict(request: PredictRequest):
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        # Fast rejection avoids image decoding after handoff. The identical
        # check under _MODEL_LOCK below is the authoritative race-free gate.
        with _ACTIONS_LOCK:
            try:
                _require_matching_binding(request.binding_id)
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if not _ACTIONS_ENABLED:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "VLA action inference is disabled after controller handoff"
                    ),
                )
        try:
            import torch

            env_obs = build_env_observation(request.model_dump())
            with _MODEL_LOCK:
                with _ACTIONS_LOCK:
                    try:
                        _require_matching_binding(request.binding_id)
                    except ValueError as error:
                        raise HTTPException(
                            status_code=409, detail=str(error)
                        ) from error
                    if not _ACTIONS_ENABLED:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "VLA action inference is disabled after controller "
                                "handoff"
                            ),
                        )
                with torch.no_grad():
                    actions, _ = _MODEL.predict_action_batch(
                        env_obs,
                        mode="eval",
                        compute_values=False,
                    )
            if torch.is_tensor(actions):
                actions = actions.detach().float().cpu().numpy()
            actions = np.asarray(actions, dtype=np.float32)
            if (
                actions.ndim != 3
                or actions.shape[0] != 1
                or actions.shape[2] != ACTION_DIM
                or actions.shape[1] < 1
                or actions.shape[1] > DEFAULT_ACTION_CHUNK
            ):
                raise ValueError(
                    f"Pi0.5 returned invalid [1,T,23] shape {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("Pi0.5 returned NaN or infinity")
            return {
                "actions": actions.tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            }
        except HTTPException:
            raise
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.exception("Pi0.5 prediction failed")
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"},
                status_code=500,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help=(
            "Shared BEHAVIOR Pi0.5 checkpoint. Other checkpoints, including "
            "task-specific SFT checkpoints, are rejected."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_model(args.checkpoint, seed=args.seed)

    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
