from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
import numpy as np
import pytest

from robots.behavior.env_server import BehaviorEnvFacade


def test_pure_vla_runtime_requires_one_complete_32_by_23_chunk():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._active_vla_invocation = {"invocation_id": "vla-1"}
    facade._official_success_latched = False
    facade._last_info = {"done": {"success": False}}
    facade._controller_state = "vla"
    facade._vla_actions_enabled = True

    with pytest.raises(ValueError, match=r"complete \[32,23\] chunk"):
        facade.pi0_nav_pick_chunk_step(
            np.zeros((31, 23), dtype=np.float32),
            chunk_index=1,
        )
