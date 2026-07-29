from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
from robots.behavior.schemas import PI0_NAV_PICK_SPEC


def test_pure_vla_public_contract_is_one_complete_chunk_sequence():
    schema = PI0_NAV_PICK_SPEC["input_schema"]
    assert schema["required"] == ["instruction", "chunks"]
    assert set(schema["properties"]) == {"instruction", "chunks"}
    assert schema["properties"]["chunks"]["minimum"] == 1
