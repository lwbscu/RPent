# BEHAVIOR prompt and evaluation organization

## Purpose

This plugin runs the BEHAVIOR-1K 2025 challenge environment with an R1Pro,
Pi0.5, and an LLM planner. It follows the same RPent process boundary as
LIBERO: the RPent process owns the cerebrum and toolkit, `env_server.py` owns
OmniGibson, and `vla_server.py` owns Pi0.5.

## Start from an up-to-date fork

Reproduction should start from a clean fork synchronized to current RPent
`main`, then use a committed feature revision. Do not evaluate an uncommitted
worktree.

```bash
git clone https://github.com/YOUR_ACCOUNT/RPent.git
cd RPent
git remote add upstream https://github.com/RLinf/RPent.git
git fetch --prune upstream origin
git checkout main
git merge --ff-only upstream/main
git push origin main
git checkout YOUR_BEHAVIOR_COMMIT_OR_BRANCH
git status --short             # must be empty
```

## Active prompt flow

1. `rpent/cli/main.py` renders the environment prompt and starts a fresh
   runtime.
2. `prompt_bundle.py` selects the control-mode fragments in `prompts/`.
3. `toolkit.py` exposes only the tools allowed by that mode.
4. `runtime_provider.py` launches and owns fresh env/VLA process groups.
5. `env_server.py` validates camera lineage, checkpoint provenance, CuRobo
   motion, grasp state, and raw official success.

`--behavior-stage3-press` is an explicit opt-in. It is accepted only for
`pi0_nav_pick_vla` on `turning_on_radio`; without it, the public surface stops
at the reviewed pre-press checkpoint.

## Checkpoint policy

- The only persistent checkpoint JSON files are `state_checkpoint_1.json`
  (post-pick) and `state_checkpoint_2.json` (geometry-verified pre-press).
- The LLM may create at most four same-run, non-overwritable
  `tmp_state_checkpoint_<label>.json` recovery points.
- Every restore is a guarded CuRobo robot motion from run-bound JSON. Scene,
  simulator, tensor-state, and snapshot restore paths are not part of the
  public runtime.
- Temporary checkpoint JSON files are deleted when the runtime ends. A serial
  evaluation is invalid if one remains.

## One Codex SDK run

Configure the external BEHAVIOR/RLinf environment and Pi0.5 checkpoint, then
run a fresh owned process stack:

```bash
export RPENT_RLINF_ROOT=/path/to/RLinf_agentic_push
export PI05_CHECKPOINT_PATH=/path/to/pi05-behavior-checkpoint

python -m rpent.cli.main \
  --env behavior --cerebrum codex --model YOUR_CODEX_MODEL \
  --behavior-control-mode pi0_nav_pick_vla --behavior-stage3-press \
  --task 0 --task-name turning_on_radio \
  --activity-instance-id 242 --seed 0 --cuda-device 0 \
  --behavior-pi0-pick-instruction \
  "Turn on the radio receiver that's on the table in the living room." \
  --output-dir /path/to/new-empty-run-dir
```

Do not add `--no-driver`, reuse an endpoint, resume another run, or issue a
second navigation/grasp call. `final_result.json.task_success` is accepted only
when its source is raw `info["done"]["success"]`; process exit code, local grasp,
checkpoint creation, contact, and LLM prose are not task success.

## Strictly serial public evaluation

The official self-evaluation slice is the first ten ordered IDs in BEHAVIOR's
authoritative `test_instances.csv`. The last ten form a holdback slice. The
runner verifies the CSV hash and order, freezes all argv before the first
episode, starts a new RPent/Codex/env/VLA process for each ID, permits one
attempt, never retries, and never adapts a later command from an earlier
result.

```bash
python scripts/run_behavior_serial_eval.py \
  --split official_first10 \
  --cuda-device 0 \
  --model YOUR_CODEX_MODEL \
  --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
  --output-root /path/to/new-empty-eval-root
```

The root contains `eval_plan.json`, `eval_results.jsonl`, and
`eval_summary.json`. A `passed` entry requires bound manifests, clean runtime
shutdown, raw official success, the post-success hold, and a fresh final
press-wrist image for green-marker review. This binary public-slice result is
not the complete BEHAVIOR challenge leaderboard score.

## Validation

Before committing or evaluating:

```bash
ruff format robots/behavior scripts/run_behavior_serial_eval.py tests
ruff check robots/behavior scripts/run_behavior_serial_eval.py tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q robots/behavior rpent
```

Prompt changes must keep dynamic `held_hand` / `press_hand` binding, fresh
frame/projection lineage, full press-gripper closure, raw-success semantics,
and temporary-checkpoint cleanup explicit.
