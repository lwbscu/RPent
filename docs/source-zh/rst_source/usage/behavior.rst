BEHAVIOR
========

BEHAVIOR 插件在 BEHAVIOR-1K 2025 challenge 中运行 R1Pro。RPent 主进程持有
LLM 与工具循环，``env_server.py`` 持有 OmniGibson，``vla_server.py`` 持有
Pi0.5 权重。一次正式评测必须新建并完整清理这组进程，不能复用常驻 endpoint。

准备代码与运行环境
------------------

先让自己的 fork 与 RPent 最新 ``main`` 同步，再切到已提交的 BEHAVIOR 版本；
正式评测不接受脏 worktree。

.. code-block:: bash

   git remote add upstream https://github.com/RLinf/RPent.git
   git fetch --prune upstream origin
   git checkout main
   git merge --ff-only upstream/main
   git push origin main

   export RPENT_RLINF_ROOT=/path/to/RLinf_agentic_push
   export PI05_CHECKPOINT_PATH=/path/to/pi05-behavior-checkpoint

单个 Codex SDK episode
----------------------

.. code-block:: bash

   python -m rpent.cli.main \
     --env behavior --cerebrum codex --model YOUR_CODEX_MODEL \
     --behavior-control-mode pi0_nav_pick_vla --behavior-stage3-press \
     --task 0 --task-name turning_on_radio \
     --activity-instance-id 242 --seed 0 --cuda-device 0 \
     --output-dir /path/to/new-empty-run-dir

Stage 3 开关只允许用于 ``turning_on_radio`` 的 ``pi0_nav_pick_vla`` 模式；
它在 post-pick/pre-press 工具面之后增加有界的按钮接触 handler，但 LLM 仍只能
调用一次 ``pi0_nav_pick``。

Checkpoint 与成功规则
----------------------

长期保留的 JSON 只有 ``state_checkpoint_1.json`` 与
``state_checkpoint_2.json``。LLM 可自主创建最多四个同 run、不可覆盖的
``tmp_state_checkpoint_<label>.json`` 恢复点。恢复始终读取本 run JSON，经过
CuRobo 规划并执行机器人运动；不得加载 simulator 或 scene state。run 结束时
runtime 必须删除全部临时 checkpoint JSON。

官方成功只认 ``final_result.json.task_success=true``，且来源必须是原始
``info["done"]["success"]``。退出码 0、局部抓取、保存 checkpoint、接触或 LLM
文字都不能替代它。成功按压还必须在 render hold 后保留 fresh press-wrist 图片，
供绿点审核。

串行 public instance 评测
-------------------------

单卡环境使用仓库内的严格串行 runner：

.. code-block:: bash

   python scripts/run_behavior_serial_eval.py \
     --split official_first10 --cuda-device 0 \
     --model YOUR_CODEX_MODEL \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --output-root /path/to/new-empty-eval-root

runner 按 authoritative CSV 原顺序使用前十个 instance，先固化不可变计划，随后
每个 instance 启动全新的 RPent/Codex/env/VLA 顶层进程；每个只允许一次尝试，
没有自动重试，也不会根据上一实例调整下一实例命令。输出为
``eval_plan.json``、``eval_results.jsonl`` 和 ``eval_summary.json``。脏源码、
非空目录、ID 重排、残留子进程或残留临时 checkpoint 都会使结果失效。这是
RPent 的 public slice 二值评测，不等同于完整 challenge leaderboard scorer。

prompt 的文件职责与 contributor 检查见 ``robots/behavior/README.md``。
