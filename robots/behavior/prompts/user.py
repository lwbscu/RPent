"""User prompt section bodies for one BEHAVIOR public cell."""

from __future__ import annotations

CELL = """- task: {{ task_name }}
- authoritative task language: {{ task_language }}
- phase: {{ behavior_phase }}
- public seed: {{ public_seed }}
- tag: {{ recipe_tag }}
- output root: {{ output_dir }}"""

BEGIN = """{{ behavior_user_instructions }}"""
