#!/usr/bin/env python3
"""Diagnostic-only R1Pro BASE cuRobo matrix; never executes a trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.behavior.env_server import BehaviorEnvFacade, _load_env_config
from robots.behavior.planner_executor import _jsonable


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--activity-instance-id", type=int, default=211)
    parser.add_argument("--compatible-only", action="store_true")
    parser.add_argument("--execute-known-navigation", action="store_true")
    parser.add_argument("--probe-arm-candidates", action="store_true")
    return parser.parse_args()


def _target_pose_from_base_joints(robot: Any, xyyaw: np.ndarray):
    import torch
    from omnigibson.utils import transform_utils as transform

    base_joints = robot.get_joint_positions()[robot.base_idx]
    pos = torch.tensor(
        [float(xyyaw[0]), float(xyyaw[1]), float(base_joints[2])],
        dtype=torch.float32,
    )
    euler = torch.tensor(
        [float(base_joints[3]), float(base_joints[4]), float(xyyaw[2])],
        dtype=torch.float32,
    )
    return pos, transform.mat2quat(transform.euler_intrinsic2mat(euler))


def _summarize_results(results: Any) -> dict[str, Any]:
    if isinstance(results, tuple):
        successes, _paths = results
        return {"successes": _jsonable(successes)}
    return {
        "batches": [
            {
                "success": _jsonable(result.success),
                "status": str(getattr(result, "status", "unavailable")),
                "valid_query": _jsonable(
                    getattr(result, "valid_query", "unavailable")
                ),
            }
            for result in results
        ]
    }


def _probe_call(
    generator: Any,
    *,
    robot: Any,
    emb_sel: Any,
    name: str,
    target_xyyaw: np.ndarray,
    ik_only: bool,
    world_collision: bool,
    exact_starter_budget: bool = False,
) -> dict[str, Any]:
    import torch

    pos, quat = _target_pose_from_base_joints(robot, target_xyyaw)
    batch_size = int(generator.batch_size)
    target_pos = {
        robot.base_footprint_link_name: torch.stack([pos] * batch_size)
    }
    target_quat = {
        robot.base_footprint_link_name: torch.stack([quat] * batch_size)
    }
    started = time.monotonic()
    try:
        results = generator.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=(math.ceil(100 / batch_size) if exact_starter_budget else 5),
            timeout=(60.0 if exact_starter_budget else 10.0),
            ik_fail_return=(50 if exact_starter_budget else 5),
            enable_finetune_trajopt=not ik_only,
            finetune_attempts=(1 if not ik_only else 0),
            return_full_result=True,
            success_ratio=1.0 / batch_size,
            attached_obj=None,
            attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=True,
            ik_only=ik_only,
            ik_world_collision_check=world_collision,
            emb_sel=emb_sel,
        )
        outcome = {"ok": True, **_summarize_results(results)}
    except Exception as exc:
        outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": name,
        "target_xyyaw": target_xyyaw.tolist(),
        "target_position": _jsonable(pos),
        "target_quaternion_xyzw": _jsonable(quat),
        "ik_only": ik_only,
        "world_collision": world_collision,
        "exact_starter_budget": exact_starter_budget,
        "elapsed_s": round(time.monotonic() - started, 3),
        **outcome,
    }


def main() -> None:
    args = _args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_args = SimpleNamespace(
        suite="behavior_2025_challenge",
        task=0,
        task_name="turning_on_radio",
        activity_definition_id=0,
        activity_instance_id=args.activity_instance_id,
        activity_instance_dir=args.activity_instance_dir,
        scene_model="house_double_floor_lower",
        seed=args.seed,
        max_episode_steps=24756,
        output_dir=str(output_dir),
        config_path=None,
        control_mode="planner_tools",
    )
    meta = vars(env_args).copy()
    env = BehaviorEnvFacade(
        cfg=_load_env_config(env_args),
        meta=meta,
        output_dir=output_dir,
        control_mode="planner_tools",
    )
    report_path = output_dir / "base_curobo_probe.json"
    report: dict[str, Any] = {
        "status": "initializing",
        "cases": [],
        "process": {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "ports": [],
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[1]),
                    "status",
                    "--porcelain",
                ],
                text=True,
            ).strip()
        ),
        "configuration": meta,
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        env.reset()
        backend = env._planner.backend
        robot = backend._find_robot()
        current = backend.get_base_pose()
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
        from omnigibson.action_primitives.curobo import (
            CuRoboEmbodimentSelection,
            CuRoboMotionGenerator,
        )
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            m as starter_macros,
        )

        emb_sel = CuRoboEmbodimentSelection.BASE
        if args.compatible_only:
            generator = backend._generator(kind="base")
            report["generator_mode"] = "rpent_scene_workspace_official_base"
            report["base_prismatic_workspace_limit_m"] = (
                backend._base_workspace_limit_m
            )
        else:
            generator = CuRoboMotionGenerator(
                robot=robot,
                batch_size=3,
                collision_activation_distance=(
                    starter_macros.DEFAULT_COLLISION_ACTIVATION_DISTANCE
                ),
            )
            report["generator_mode"] = "official_starter_exact"
        generator.update_obstacles()
        kinematics = generator.mg[emb_sel].kinematics
        report.update(
            {
                "status": "running",
                "current_base_world_xyyaw": current.tolist(),
                "base_idx": _jsonable(robot.base_idx),
                "base_control_idx": _jsonable(robot.base_control_idx),
                "current_base_joint_values": _jsonable(
                    robot.get_joint_positions()[robot.base_idx]
                ),
                "robot_joint_names": list(generator.robot_joint_names),
                "base_link": str(generator.base_link[emb_sel]),
                "eef_link": str(generator.ee_link[emb_sel]),
                "kinematics_joint_names": list(kinematics.joint_names),
                "kinematics_joint_limits": _jsonable(
                    kinematics.kinematics_config.joint_limits.position
                ),
                "current_q_shape": list(q.shape),
            }
        )
        save()
        targets = [
            ("current_ik_collision", current.copy(), True, True, False),
            ("current_ik_no_world_collision", current.copy(), True, False, False),
            ("x_plus_5cm_ik_collision", current + [0.05, 0.0, 0.0], True, True, False),
            ("y_plus_5cm_ik_collision", current + [0.0, 0.05, 0.0], True, True, False),
            ("yaw_plus_3deg_ik_collision", current + [0.0, 0.0, math.radians(3)], True, True, False),
            ("x_plus_5cm_full", current + [0.05, 0.0, 0.0], False, True, False),
            ("yaw_plus_3deg_full", current + [0.0, 0.0, math.radians(3)], False, True, False),
            (
                "known_candidate_exact_starter_full",
                np.asarray([5.6047677074, 5.1226747401, -0.2617993878]),
                False,
                True,
                True,
            ),
        ]
        if args.compatible_only:
            targets = [
                target
                for target in targets
                if target[0]
                in {
                    "current_ik_collision",
                    "x_plus_5cm_ik_collision",
                    "x_plus_5cm_full",
                    "yaw_plus_3deg_full",
                    "known_candidate_exact_starter_full",
                }
            ]
            targets[-1] = (*targets[-1][:-1], False)
        for name, target, ik_only, world_collision, exact_budget in targets:
            report["cases"].append(
                _probe_call(
                    generator,
                    robot=robot,
                    emb_sel=emb_sel,
                    name=name,
                    target_xyyaw=np.asarray(target, dtype=np.float64),
                    ik_only=ik_only,
                    world_collision=world_collision,
                    exact_starter_budget=exact_budget,
                )
            )
            save()
        if args.probe_arm_candidates:
            surface = np.asarray([6.4258046597, 4.9026785517, 0.9806372991])
            known_candidate = np.asarray([5.6047677074, 5.1226747401, -0.2617993878])
            direction = known_candidate[:2] - surface[:2]
            direction /= np.linalg.norm(direction)
            report["arm_candidate_probes"] = []
            probe_index = 0
            for standoff_m in (0.85, 0.75, 0.65):
                candidate = np.asarray(
                    [
                        surface[0] + direction[0] * standoff_m,
                        surface[1] + direction[1] * standoff_m,
                        math.atan2(-direction[1], -direction[0]),
                    ]
                )
                for clearance_m in (0.08, 0.15, 0.25):
                    target = backend._candidate_reachability_target(
                        surface,
                        candidate,
                        clearance_m=clearance_m,
                    )
                    started = time.monotonic()
                    reachable, reason, metrics = backend.check_arm_reachability(
                        hand="left",
                        target_xyz=target,
                        target_quat_xyzw=None,
                        base_xyyaw=candidate,
                        timeout_s=8.0,
                        skip_obstacle_update=probe_index > 0,
                    )
                    probe_index += 1
                    report["arm_candidate_probes"].append(
                        {
                            "standoff_m": standoff_m,
                            "clearance_m": clearance_m,
                            "candidate_xyyaw": candidate.tolist(),
                            "target_xyz": target.tolist(),
                            "reachable": reachable,
                            "reason": reason,
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "metrics": metrics,
                        }
                    )
                    save()
        if args.execute_known_navigation:
            report["navigation_execution"] = env.navigate_to(
                hand="left",
                target_xyz=[6.4258046597, 4.9026785517, 0.9806372991],
                standoff_m=0.85,
                timeout_s=180.0,
            )
            save()
        report["status"] = "complete"
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
