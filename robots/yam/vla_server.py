# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""YAM Pi0.5 RPC adapter; imports the existing RLinf YAM policy transforms."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np

from robots.yam.contracts import MODEL_SPEC, validate_actions, vla_runtime_contract
from rpent.robots.components.vla_facade_base import BaseVLAFacade


def build_model_cfg(model_path: str, norm_stats_path: str | None = None):
    from omegaconf import OmegaConf

    data = {"norm_stats_path": norm_stats_path} if norm_stats_path else {}
    return OmegaConf.create({
        "model_type": "openpi",
        "model_path": model_path,
        "precision": None,
        "num_action_chunks": MODEL_SPEC.use_length,
        "action_dim": 14,
        "is_lora": False,
        "lora_rank": 32,
        "use_proprio": True,
        "num_steps": 5,
        "add_value_head": False,
        "openpi_data": data,
        "openpi": {
            "config_name": MODEL_SPEC.policy_name,
            "num_images_in_input": 3,
            "action_horizon": MODEL_SPEC.action_horizon,
            "action_chunk": MODEL_SPEC.use_length,
            "action_env_dim": 14,
            "num_steps": 5,
            "noise_level": 0.5,
            "noise_method": "flow_sde",
            "train_expert_only": True,
            "add_value_head": False,
            "value_after_vlm": False,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": None,
            "use_dsrl": False,
        },
    })


class YamVLAFacade(BaseVLAFacade):
    def __init__(
        self, model=None, *, model_path=None, norm_stats_path=None, device="cuda"
    ):
        self._inference_context = contextlib.nullcontext
        if model is None:
            if not model_path:
                raise ValueError(
                    "provide a trained YAM model_path, including YAM norm_stats"
                )
            root = os.environ.get("RPENT_RLINF_ROOT") or os.environ.get(
                "RLINF_REPO_PATH"
            )
            if root:
                root = str(Path(root).expanduser().resolve())
                sys.path.insert(0, root)
            import torch
            from rlinf.models.embodiment.openpi import get_model

            model = get_model(
                build_model_cfg(model_path, norm_stats_path), torch_dtype=None
            )
            model = model.to(device).eval()
            self._inference_context = torch.inference_mode
        self._model = model
        super().__init__()
        self._rpc["vla.get_meta"] = vla_runtime_contract
        self._readonly_methods.add("vla.get_meta")

    def predict(self, observation, options=None):
        options = options or {}
        if not isinstance(options, dict) or set(options) - {"mode"}:
            raise ValueError("supported VLA options: mode='eval'")
        if options.get("mode", "eval") != "eval":
            raise ValueError("YAM deployment accepts only eval inference")
        if not isinstance(observation, dict):
            raise TypeError("observation must be a mapping")
        top = np.asarray(observation.get("main_images"))
        side = np.asarray(observation.get("extra_view_images"))
        states = np.asarray(observation.get("states"), dtype=np.float32)
        if (
            top.ndim != 4
            or top.shape[0] != 1
            or top.shape[-1] != 3
            or top.dtype != np.uint8
        ):
            raise ValueError("main_images must be uint8 RGB [1,H,W,3]")
        if side.shape != (1, 2, *top.shape[1:]) or side.dtype != np.uint8:
            raise ValueError(
                "extra_view_images must be uint8 RGB [1,2,H,W,3], ordered left/right"
            )
        if states.shape != (1, 14) or not np.isfinite(states).all():
            raise ValueError("states must be finite [1,14]")
        descriptions = observation.get("task_descriptions")
        if (
            not isinstance(descriptions, list)
            or len(descriptions) != 1
            or not isinstance(descriptions[0], str)
            or not descriptions[0].strip()
        ):
            raise ValueError("task_descriptions must contain one nonempty instruction")
        env_obs = {
            "main_images": top,
            "extra_view_images": side,
            "wrist_images": None,
            "states": states,
            "task_descriptions": descriptions,
        }
        with self._inference_context():
            actions, _ = self._model.predict_action_batch(env_obs, mode="eval")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"policy output must be [1,T,14]; got {actions.shape}")
        result = validate_actions(actions[0])
        if len(result) > MODEL_SPEC.use_length:
            raise ValueError(
                f"policy output exceeds configured use_length={MODEL_SPEC.use_length}"
            )
        return result[None].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8220)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    facade = YamVLAFacade(
        model_path=args.model_path,
        norm_stats_path=args.norm_stats_path,
        device=args.device,
    )
    facade.serve(
        host=args.host,
        port=args.port,
        transport=args.transport,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
