# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""User prompt for one YAM run."""

CELL = """- task: {{task_name}}
- seed label: {{seed}}
- instruction: {{instruction}}
- policy: pi05_yam_joint qpos14
"""

BEGIN = """Follow the required read order, bind the current target and relation
from fresh YAM observations, then execute the first unmet phase. Use the complete
task language as the overall goal. For pi05_act you may supply a trained,
task-relevant subgoal via prompt; do not replace the episode goal.
Verify env eval_success before finish."""
