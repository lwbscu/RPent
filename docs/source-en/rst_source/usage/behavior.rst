BEHAVIOR
========

The BEHAVIOR plugin runs the R1Pro in BEHAVIOR-1K's 2025 challenge tasks.
RPent owns the LLM/tool loop, ``env_server.py`` owns OmniGibson, and
``vla_server.py`` owns the Pi0.5 weights. The processes are launched and
cleaned as one run; formal evaluation must not reuse resident endpoints.

Preparation
-----------

Start from a clean fork synchronized with current RPent ``main`` and check out
the committed BEHAVIOR revision. Configure the external RLinf/BEHAVIOR checkout
and Pi0.5 checkpoint:

.. code-block:: bash

   git remote add upstream https://github.com/RLinf/RPent.git
   git fetch --prune upstream origin
   git checkout main
   git merge --ff-only upstream/main
   git push origin main

   export RPENT_RLINF_ROOT=/path/to/RLinf_agentic_push
   export PI05_CHECKPOINT_PATH=/path/to/pi05-behavior-checkpoint
   export RPENT_PYTHON=/path/to/RPent/.venv/bin/python

One Codex SDK episode
---------------------

.. code-block:: bash

   "$RPENT_PYTHON" -m rpent.cli.main \
     --env behavior --cerebrum codex --model YOUR_CODEX_MODEL \
     --behavior-control-mode pi0_nav_pick_vla --behavior-stage3-press \
     --task 0 --task-name turning_on_radio \
     --activity-instance-id 242 --seed 0 --cuda-device 0 \
     --output-dir /path/to/new-empty-run-dir

The stage-3 flag is valid only for ``turning_on_radio`` in
``pi0_nav_pick_vla`` mode. It extends the normal post-pick/pre-press surface
with bounded button-contact handlers. The LLM still calls ``pi0_nav_pick``
exactly once.

Checkpoint and success rules
----------------------------

Only ``state_checkpoint_1.json`` and ``state_checkpoint_2.json`` persist.
The LLM may create at most four non-overwritable, same-run
``tmp_state_checkpoint_<label>.json`` recovery points. Restore always plans and
executes a guarded CuRobo robot motion; it never loads simulator or scene
state. The runtime deletes every temporary checkpoint JSON at run end.

Official success is only ``final_result.json.task_success=true`` sourced from
raw ``info["done"]["success"]``. Exit code zero, local grasp, checkpoint save,
contact, or model text cannot replace it. A successful press also preserves a
fresh press-wrist image after the render-synchronization hold for green-marker
review.

Serial public-instance evaluation
---------------------------------

Use the checked-in runner when one GPU cannot host parallel simulations:

.. code-block:: bash

   "$RPENT_PYTHON" scripts/run_behavior_serial_eval.py \
     --split official_first10 --cuda-device 0 \
     --model YOUR_CODEX_MODEL \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --output-root /path/to/new-empty-eval-root

The runner uses the first ten IDs in the authoritative CSV in their original
order, freezes an immutable plan, runs one fresh top-level process per
instance, and permits no automatic retry. It records ``eval_plan.json``,
``eval_results.jsonl``, and ``eval_summary.json``. It refuses dirty source,
nonempty run directories, reordered IDs, lingering managed processes, and
temporary-checkpoint residue. Before plan creation it also verifies that the
frozen RPent Python imports HTTPX, the Codex SDK, the CLI, and the BEHAVIOR
runtime. This is RPent's binary public-slice evaluation, not the full challenge
leaderboard scorer.

See ``robots/behavior/README.md`` for prompt ownership and contributor checks.
