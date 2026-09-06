# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pi0.5 client with the YAM training/deployment view mapping."""

import numpy as np

from robots.yam.contracts import MODEL_SPEC, validate_actions, vla_runtime_contract
from rpent.robots.components.vla_client_base import BaseVLAClient


def policy_observation(
    snapshot: dict, instruction: str, *, prompt: str | None = None
) -> dict:
    frames = snapshot["frames"]
    policy_instruction = (
        prompt if prompt is not None and prompt.strip() else instruction
    )
    return {
        "main_images": np.asarray(frames["top"])[None],
        "wrist_images": None,
        "extra_view_images": np.stack([frames["left"], frames["right"]])[None],
        "states": np.asarray(snapshot["state"]["joint_position"], dtype=np.float32)[
            None
        ],
        "task_descriptions": [policy_instruction],
    }


class YamVLAClient(BaseVLAClient):
    def __init__(self, client):
        super().__init__(client)
        actual = client.call("vla.get_meta", timeout_s=30)
        expected = vla_runtime_contract()
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"vla.get_meta.{key}: expected {value!r}, got {actual.get(key)!r}"
                )

    def predict(self, obs, options=None):
        actions = np.asarray(super().predict(obs, options))
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = validate_actions(actions)
        if len(actions) > MODEL_SPEC.use_length:
            raise ValueError(
                f"VLA returned {len(actions)} actions; use_length={MODEL_SPEC.use_length}"
            )
        return actions

    infer = predict
