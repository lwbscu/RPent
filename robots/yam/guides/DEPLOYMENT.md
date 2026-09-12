# YAM: Pi0.5 and supervised exploration

Use the same RPent revision on both hosts. The control host keeps its RLinf
checkout, calibration, SDK tuning, and recorded poses. The inference host uses
its existing RLinf environment and checkpoint; no separate model transforms are
installed. The Agent uses RPent's environment. These are source-checkout commands.

## Site configuration

Start from `robots/yam/config.example.json` and the station's validated geometry
configuration. Supply actual camera serials, extrinsics, finite table geometry,
collision/servo settings, and an operator receipt path. Add the station's
confirmed `reset` and `park_on_close` blocks from the RLinf evaluation YAML;
each contains `enabled`, seven-element `left_qpos` and `right_qpos`, `duration_s`,
`max_joint_delta`, `tolerance`, and `timeout_s`. Home must be enabled for the
formal environment service. Do not copy another station's joint poses.

This deployment uses these independent site files under
`/home/yambox/cynws/RPent/logs/yambox_deployment/site/`:

| File / task ID | Left bag | Right bag | Seed label |
| --- | --- | --- | --- |
| `task_a_explore.json` / `tabletop_cleanup_a` | Pepsi | Coca-Cola | 0 |
| `task_b_explore.json` / `tabletop_cleanup_b` | Coca-Cola | Pepsi | 1 |

Both tasks use the top camera's left/right convention. All three bottles go in
the assigned bags. White spoon goes in the left white bowl; pink spoon goes in
the right pink bowl. Move obstructing bowls and return them to their marks.
The seed labels identify runs; they do not randomize or restore the real scene.
The initial supervised deployment has 1,800 executed control steps per attempt,
matching the RLinf evaluation budget. Operator waiting does not consume steps.

## Inference host: start VLA

```bash
ssh zxcx@192.168.120.63
cd /home/zxcx/cynws/RPent
source /home/zxcx/cynws/RLinf/.venv/bin/activate
export RPENT_RLINF_ROOT=/home/zxcx/cynws/RLinf
python -m robots.yam.vla_server \
  --model-path /home/zxcx/ckpt/yam \
  --norm-stats-path /home/zxcx/ckpt/yam/norm_stats.json \
  --transport socket --host 0.0.0.0 --port 8220
```

The public RPC client checks the model, RGB camera order, absolute qpos14 layout,
and 30-step horizon. Use these pickle RPC endpoints on the trusted robot LAN.
Loading VLA alone does not connect robot hardware.

## Control host: environment and Agent

Select A in each control-host terminal (for B replace `a` with `b`):

```bash
cd /home/yambox/cynws/RPent
source .venv/bin/activate
export RPENT_RLINF_ROOT=/home/yambox/cynws/RLinf
export YAM_SITE="$PWD/logs/yambox_deployment/site/task_a_explore.json"
```

In the environment terminal, with the operator present and startup/gripper
calibration paths clear:

```bash
python -m robots.yam.env_server --config "$YAM_SITE" \
  --transport socket --host 127.0.0.1 --port 8110
```

In the Agent terminal, after the operator has prepared the scene:

```bash
rpent --robot yam --planner codex --explore \
  --env-endpoint socket://127.0.0.1:8110 \
  --vla-endpoint socket://192.168.120.63:8220 \
  --task-name "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_name"])' "$YAM_SITE")" \
  --task-language "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_language"])' "$YAM_SITE")" \
  --seed "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$YAM_SITE")" \
  --max-episode-steps 1800 \
  --explore-sessions 1 --explore-attempts-per-session 5 \
  --memory-profile local --memory-dir "$PWD/memory/yam" \
  --auto-merge-memory --planner-timeout-s 3600 --max-turns 300
```

The first observation starts the cameras and robot connection. Check Agent
connectivity without hardware separately with `rpent-check-llm --planner codex`.
Codex must have a working local login/API configuration and a compatible CLI.

## Operator terminal

Use the same setup and `YAM_SITE`. Define this shortcut once:

```bash
yam_operator() {
  python -m robots.yam.operator_control --config "$YAM_SITE" \
    --endpoint socket://127.0.0.1:8110 "$@"
}
yam_operator --event status
```

With a clear return path, move to the recorded start pose:

```bash
yam_operator --event reset_pose
```

Read `status`, copy its **current** `episode_id`, and confirm the restored scene:

```bash
yam_operator --event ready --episode-id CURRENT_ID --note 'Scene restored; ready.'
```

The Agent consumes ready through its `reset` tool, which creates a new episode
ID. Read status again before giving a verdict; do not reuse the ready ID:

```bash
yam_operator --event status
yam_operator --event success --episode-id CURRENT_ID --note 'All task relations verified.'
# Or, for an unsuccessful attempt:
yam_operator --event failure --episode-id CURRENT_ID --note 'Describe the observed failure.'
# Stop the session:
yam_operator --event abort --episode-id CURRENT_ID --note 'Operator abort.'
```

After failure the Agent records its evidence and waits. Restore objects, use
`reset_pose` only with a clear path, then issue ready for the currently advertised
ID. Pending waits do not consume attempts. After success the Agent distils the
lesson and stops; no further retry is required. Five failed attempts exhaust the
session. The Agent must never issue or fabricate operator receipts.

After the planner stops, the environment still holds the arms. Once the home
path is clear, close it explicitly:

```bash
yam_operator --event shutdown
```

This returns home before disabling output. If home fails, keep the process alive,
inspect the obstruction/posture, and retry. Switch A/B only after closing the old
environment; restart it with the new site file. Do not leave a task-A service
running behind task-B CLI arguments.

## Evidence and reuse

The run directory contains official EnvState observations and action records,
one MP4 per attempt, result/transcript files, and the current successful episode's
`*_recipe.jsonl` / audit JSON. MP4 contains executed top frames; long planner or
operator pauses are not represented as real-time idle video.

The common MemoryManager owns `memory/yam/MEMORY.md`, `task_only`, `suite`,
`global`, and `_internal/inbox`. It merges the current unique inbox after a run;
the next run reads the index through common file tools. A task-specific lesson
must keep its A/B task ID and bag rule. Shared perception/motion lessons can be
global without imposing either bag mapping. Existing conflict archives remain
under the official manager's control. An interface smoke test is not evidence
that the physical task succeeded; success requires that episode's operator
receipt and the resulting recipe/audit and memory artifacts.
