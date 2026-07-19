"""Environment-side BEHAVIOR planner primitives backed by RGB-D and cuRobo."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.camera_geometry import (
    CameraGeometryError,
    FrameCache,
    backproject_pixel_to_world,
    canonical_camera,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    ENV_ACTION_SEGMENTS,
    validate_action_chunk,
)

LEFT_EEF_LINK = "left_eef_link"
RIGHT_EEF_LINK = "right_eef_link"
EEF_LINK_BY_HAND = {"left": LEFT_EEF_LINK, "right": RIGHT_EEF_LINK}
GRIPPER_COMMAND_BY_OPENING = {"open": 1.0, "closed": -1.0}


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def official_task_success(info: Any) -> bool:
    """Read only BEHAVIOR's official success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    return bool(done.get("success", False)) if isinstance(done, dict) else False


def primitive_result(
    *,
    primitive_success: bool,
    task_success: bool,
    stop_reason: str,
    recoverable: bool,
    suggested_next_tool: str | None = None,
    metrics: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared planner result envelope."""
    return {
        "primitive_success": bool(primitive_success),
        "task_success": bool(task_success),
        "official_success_source": 'info["done"]["success"]',
        "stop_reason": str(stop_reason),
        "recoverable": bool(recoverable),
        "suggested_next_tool": suggested_next_tool,
        "metrics": _jsonable(metrics or {}),
        "diagnostics": _jsonable(diagnostics or {}),
    }


def _normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value not in EEF_LINK_BY_HAND:
        raise ValueError("hand must be 'left' or 'right'")
    return value


def _as_xyz(value: Any, *, name: str = "target_xyz") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _quat_xyzw(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError("quaternion must contain exactly 4 xyzw values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8 or not math.isfinite(norm):
        raise ValueError("quaternion has zero or invalid norm")
    return quat / norm


def _axis_angle_to_quat_xyzw(axis_angle: Any) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(-1)
    if vec.shape != (4,):
        raise ValueError("relative_axis_angle must contain [axis_x, axis_y, axis_z, angle_rad]")
    axis = vec[:3]
    angle = float(vec[3])
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        if abs(angle) <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        raise ValueError("relative_axis_angle axis cannot be zero for nonzero angle")
    axis = axis / norm
    if abs(angle) <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.concatenate([axis * math.sin(angle * 0.5), [math.cos(angle * 0.5)]])


def _quat_multiply_xyzw(a: Any, b: Any) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    out = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )
    return out / max(float(np.linalg.norm(out)), 1e-12)


def _quat_angle_error_rad(a: Any | None, b: Any | None) -> float | None:
    if a is None or b is None:
        return None
    qa = _quat_xyzw(a)
    qb = _quat_xyzw(b)
    assert qa is not None and qb is not None
    dot = abs(float(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _segment_array(value: Any) -> np.ndarray:
    array = np.asarray(_jsonable(value), dtype=np.int64).reshape(-1)
    return array


class RealCuroboBackend:
    """Lazy OmniGibson/cuRobo adapter used only inside the real env process."""

    def __init__(self, env_facade: Any, *, output_dir: str | Path | None = None) -> None:
        self.env_facade = env_facade
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._robot: Any | None = None
        self._torch: Any | None = None
        self._curobo_cls: Any | None = None
        self._embodiment_cls: Any | None = None
        self._generators: dict[str, Any] = {}
        self._config_paths: dict[str, Path] = {}
        self._attached_objects_by_hand: dict[str, Any] = {}
        self._active_generator: Any | None = None
        self._last_collision_step = -1
        self._collision_check_interval_steps = 4
        self._last_collision_report: dict[str, Any] = {
            "available": False,
            "reason": "not_checked",
            "min_margin_m": None,
        }

    def _lazy_imports(self) -> None:
        if self._curobo_cls is not None:
            return
        import torch
        from omnigibson.action_primitives.curobo import (
            CuRoboEmbodimentSelection,
            CuRoboMotionGenerator,
        )

        self._torch = torch
        self._curobo_cls = CuRoboMotionGenerator
        self._embodiment_cls = CuRoboEmbodimentSelection

    def _find_robot(self) -> Any:
        if self._robot is not None:
            return self._robot
        candidates = [
            self.env_facade,
            getattr(self.env_facade, "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_direct_process", None),
            getattr(getattr(getattr(self.env_facade, "_env", None), "_direct_process", None), "env", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            robots = getattr(candidate, "robots", None)
            if robots:
                self._robot = robots[0]
                return self._robot
            scene = getattr(candidate, "scene", None)
            robots = getattr(scene, "robots", None) if scene is not None else None
            if robots:
                self._robot = robots[0]
                return self._robot
        try:
            import omnigibson as og

            if og.sim.scenes and og.sim.scenes[0].robots:
                self._robot = og.sim.scenes[0].robots[0]
                return self._robot
        except Exception:
            pass
        raise RuntimeError("could not locate the R1Pro robot in the BEHAVIOR env")

    def _asset_curobo_dir(self, robot: Any) -> Path:
        curobo_path = getattr(robot, "curobo_path", None)
        if isinstance(curobo_path, dict):
            for value in curobo_path.values():
                path = Path(str(value))
                if path.is_file():
                    return path.parent
        elif curobo_path:
            path = Path(str(curobo_path))
            if path.is_file():
                return path.parent
        roots = [
            os.environ.get("OMNIGIBSON_ASSET_PATH"),
            os.environ.get("BEHAVIOR_ASSET_PATH"),
            os.environ.get("RPENT_RLINF_ROOT"),
            os.environ.get("RLINF_REPO_PATH"),
        ]
        for root in roots:
            if not root:
                continue
            base = Path(root).expanduser()
            candidates = [
                base
                / ".venv-behavior"
                / "BEHAVIOR-1K"
                / "datasets"
                / "omnigibson-robot-assets"
                / "models"
                / "r1pro"
                / "curobo",
                base
                / "datasets"
                / "omnigibson-robot-assets"
                / "models"
                / "r1pro"
                / "curobo",
            ]
            for candidate in candidates:
                if (candidate / "r1pro_description_curobo_arm.yaml").is_file():
                    return candidate
        raise RuntimeError("could not locate R1Pro cuRobo YAML assets")

    def _hand_config_path(self, hand: str) -> Path:
        hand = _normalize_hand(hand)
        if hand in self._config_paths:
            return self._config_paths[hand]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_arm.yaml"
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to generate hand-specific cuRobo config") from exc

        with source.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kinematics = cfg["robot_cfg"]["kinematics"]
        eef_link = self._eef_link_name(robot, hand)
        kinematics["ee_link"] = eef_link
        lock_joints = dict(kinematics.get("lock_joints") or {})
        inactive = "right" if hand == "left" else "left"
        inactive_eef_link = self._eef_link_name(robot, inactive)
        kinematics["link_names"] = [inactive_eef_link]
        for joint in [f"{inactive}_arm_joint{i}" for i in range(1, 8)]:
            self._validate_joint_name(robot, joint)
            lock_joints.setdefault(joint, None)
        for side in ("left", "right"):
            for suffix in ("finger_joint1", "finger_joint2"):
                joint = f"{side}_gripper_{suffix}"
                self._validate_joint_name(robot, joint)
                lock_joints.setdefault(joint, None)
        for joint in (
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_z_joint",
            "base_footprint_rx_joint",
            "base_footprint_ry_joint",
            "base_footprint_rz_joint",
        ):
            self._validate_joint_name(robot, joint)
            lock_joints.setdefault(joint, None)
        self._validate_lock_joint_names(robot, lock_joints)
        kinematics["lock_joints"] = dict(sorted(lock_joints.items()))
        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"r1pro_description_curobo_arm_{hand}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._config_paths[hand] = out
        return out

    def _base_config_path(self) -> Path:
        key = "base"
        if key in self._config_paths:
            return self._config_paths[key]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_base.yaml"
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to generate base cuRobo config") from exc
        with source.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kinematics = cfg["robot_cfg"]["kinematics"]
        lock_joints = dict(kinematics.get("lock_joints") or {})
        self._validate_lock_joint_names(robot, lock_joints)
        kinematics["lock_joints"] = dict(sorted(lock_joints.items()))
        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "r1pro_description_curobo_base_runtime.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._config_paths[key] = out
        return out

    def _eef_link_name(self, robot: Any, hand: str) -> str:
        expected = EEF_LINK_BY_HAND[_normalize_hand(hand)]
        eef_names = getattr(robot, "eef_link_names", None)
        if isinstance(eef_names, dict):
            value = eef_names.get(hand)
            if value:
                expected = str(value)
        links = getattr(robot, "links", {}) or {}
        if expected not in links:
            raise RuntimeError(f"R1Pro {hand} EEF link {expected!r} not found in robot links")
        return expected

    def _validate_joint_name(self, robot: Any, joint_name: str) -> None:
        joints = getattr(robot, "joints", {}) or {}
        if joint_name not in joints:
            raise RuntimeError(f"R1Pro cuRobo lock joint {joint_name!r} not found")

    def _validate_lock_joint_names(self, robot: Any, lock_joints: dict[str, Any]) -> None:
        for joint in lock_joints:
            self._validate_joint_name(robot, str(joint))

    def _probe_generator_lock_resolution(
        self,
        generator: Any,
        *,
        kind: str,
        hand: str,
        emb_sel: Any | None = None,
    ) -> None:
        """Fail early if official null lock_joints were not resolved by OG/cuRobo."""
        update = getattr(generator, "update_locked_joints", None)
        if update is not None:
            try:
                from omnigibson import lazy

                if emb_sel is None:
                    emb_sel = (
                        self._embodiment_cls.DEFAULT
                        if self._embodiment_cls is not None
                        else None
                    )
                robot = self._find_robot()
                torch = self._torch
                if torch is None:
                    import torch as torch  # type: ignore[no-redef]
                batch_size = int(getattr(generator, "batch_size", 1))
                q_pos = torch.stack([robot.get_joint_positions()] * batch_size, dim=0)
                zeros = torch.zeros_like(q_pos)
                cu_joint_state = lazy.curobo.types.state.JointState(
                    position=generator._tensor_args.to_device(q_pos),
                    velocity=generator._tensor_args.to_device(zeros),
                    acceleration=generator._tensor_args.to_device(zeros),
                    jerk=generator._tensor_args.to_device(zeros),
                    joint_names=generator.robot_joint_names,
                )
                update(cu_joint_state, emb_sel)
                kc = generator.mg[emb_sel].kinematics.kinematics_config
                positions = np.asarray(_jsonable(kc.lock_jointstate.position), dtype=np.float64)
                if not np.isfinite(positions).all():
                    raise RuntimeError("resolved lock joint positions contain NaN or infinity")
                return
            except Exception as exc:
                raise RuntimeError(
                    f"failed to verify cuRobo {kind}:{hand} null lock_joints resolution"
                ) from exc
        lock_attrs = ("lock_joints", "_lock_joints", "locked_joints", "_locked_joints")
        for attr in lock_attrs:
            if not hasattr(generator, attr):
                continue
            value = getattr(generator, attr)
            if value is None:
                continue
            flat = _jsonable(value)
            if _contains_none(flat):
                raise RuntimeError(
                    f"cuRobo {kind}:{hand} lock_joints still contain null after runtime parsing"
                )
            return

    def _generator(self, *, kind: str, hand: str = "left") -> Any:
        self._lazy_imports()
        robot = self._find_robot()
        key = f"{kind}:{hand}"
        if key in self._generators:
            return self._generators[key]
        if kind == "arm":
            robot_cfg_path: Any = str(self._hand_config_path(hand))
            use_default_embodiment_only = True
            emb_sel = self._embodiment_cls.DEFAULT
        elif kind == "base":
            robot_cfg_path = dict(getattr(robot, "curobo_path", {}) or {})
            if not robot_cfg_path:
                raise RuntimeError("R1Pro does not expose official cuRobo embodiment configs")
            emb_sel = self._embodiment_cls.BASE
            robot_cfg_path[emb_sel] = str(self._base_config_path())
            use_default_embodiment_only = False
        else:
            raise ValueError(f"unknown cuRobo generator kind {kind!r}")
        assert self._curobo_cls is not None
        generator = self._curobo_cls(
            robot,
            robot_cfg_path=robot_cfg_path,
            motion_cfg_kwargs={
                "trajopt_tsteps": 32,
                "num_trajopt_seeds": 4,
                "num_graph_seeds": 4,
                "finetune_trajopt_iters": 100,
            },
            batch_size=2,
            use_cuda_graph=False,
            use_default_embodiment_only=use_default_embodiment_only,
        )
        self._probe_generator_lock_resolution(
            generator,
            kind=kind,
            hand=hand,
            emb_sel=emb_sel,
        )
        self._generators[key] = generator
        return generator

    def get_eef_pose(self, hand: str) -> tuple[np.ndarray, np.ndarray] | None:
        robot = self._find_robot()
        link_name = EEF_LINK_BY_HAND[_normalize_hand(hand)]
        link = getattr(robot, "links", {}).get(link_name)
        if link is None:
            return None
        pos, quat = link.get_position_orientation()
        return np.asarray(_jsonable(pos), dtype=np.float64), np.asarray(_jsonable(quat), dtype=np.float64)

    def check_arm_reachability(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        base_xyyaw: np.ndarray | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        try:
            result = self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=5.0,
                ik_only=True,
                base_xyyaw=base_xyyaw,
                attached_obj=self.get_attached_object(hand),
            )
        except Exception as exc:
            return False, "planner_unavailable", {"error": f"{type(exc).__name__}: {exc}"}
        metrics = dict(result.get("metrics", {}))
        current = self.get_eef_pose(hand)
        if current is not None:
            metrics["eef_target_distance_m"] = float(np.linalg.norm(target_xyz - current[0]))
        if result.get("ok"):
            return True, "reachable_candidate", metrics
        reason = str(result.get("stop_reason", "unreachable"))
        if reason == "unreachable" and current is not None and metrics.get("eef_target_distance_m", 0.0) > 1.0:
            reason = "navigation_required"
        return False, reason, metrics

    def plan_arm_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=False,
                attached_obj=attached_obj,
            )
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": "planner_unavailable",
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ik_only": False,
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                },
            }

    def _compute_arm_plan(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        ik_only: bool,
        base_xyyaw: np.ndarray | None = None,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        generator = self._generator(kind="arm", hand=hand)
        self._active_generator = generator
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        target_quat = (
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            if target_quat_xyzw is None
            else target_quat_xyzw
        )
        planner_target = np.asarray(target_xyz, dtype=np.float64)
        robot = self._find_robot()
        initial_joint_pos = (
            self._initial_joint_pos_for_base_candidate(robot, base_xyyaw)
            if base_xyyaw is not None
            else None
        )
        if initial_joint_pos is not None:
            initial_joint_pos = torch.as_tensor(initial_joint_pos, dtype=torch.float32)
        reachability_stage = (
            "candidate_kinematic_ik_with_initial_base"
            if base_xyyaw is not None and ik_only
            else ("world_collision_ik" if ik_only else "world_collision_full_trajectory")
        )
        batch_size = max(int(generator.batch_size), 1)
        planner_targets = torch.as_tensor(
            planner_target,
            dtype=torch.float32,
        ).reshape(1, 3).repeat(batch_size, 1)
        planner_quats = torch.as_tensor(
            target_quat,
            dtype=torch.float32,
        ).reshape(1, 4).repeat(batch_size, 1)
        successes, paths = generator.compute_trajectories(
            planner_targets,
            planner_quats,
            initial_joint_pos=initial_joint_pos,
            max_attempts=5,
            timeout=min(float(timeout_s), 8.0),
            ik_fail_return=5,
            enable_finetune_trajopt=not bool(ik_only),
            finetune_attempts=0 if ik_only else 2,
            return_full_result=False,
            success_ratio=1.0 / batch_size,
            attached_obj=attached_obj,
            ik_only=bool(ik_only),
            is_local=False,
            skip_obstacle_update=False,
            ik_world_collision_check=base_xyyaw is None,
        )
        success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
        success_indices = np.flatnonzero(success_array)
        metrics = {
            "successes": success_array.tolist(),
            "ik_only": bool(ik_only),
            "is_local": False,
            "reachability_stage": reachability_stage,
            "candidate_base_xyyaw": base_xyyaw.tolist() if base_xyyaw is not None else None,
            "collision_semantics": (
                "candidate_kinematic_ik_only_post_base_recheck_required"
                if base_xyyaw is not None and ik_only
                else "world_collision_checked"
            ),
            "curobo_config": str(self._hand_config_path(hand)),
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "attached_collision_body": {"available": attached_obj is not None},
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": batch_size,
        }
        if success_indices.size == 0:
            return {
                "ok": False,
                "stop_reason": "unreachable",
                "metrics": metrics,
            }
        if ik_only:
            metrics["reachable_by_collision_free_ik"] = base_xyyaw is None
            metrics["reachable_by_candidate_kinematic_ik"] = base_xyyaw is not None
            return {"ok": True, "metrics": metrics}
        path = paths[int(success_indices[0])]
        try:
            q_traj = generator.path_to_joint_trajectory(path, get_full_js=True)
        except Exception as exc:
            raise RuntimeError(f"path_to_joint_trajectory failed: {exc}") from exc
        try:
            q_traj = _interpolate_joint_trajectory(q_traj, max_inter_dist=0.01)
        except Exception as exc:
            raise RuntimeError(f"trajectory interpolation failed: {exc}") from exc
        try:
            actions = self.q_trajectory_to_actions(q_traj, hand=hand)
        except Exception as exc:
            raise RuntimeError(f"23D action packing failed: {exc}") from exc
        try:
            collision_report = self._check_q_trajectory_collisions(
                generator,
                q_traj,
                attached_obj=attached_obj,
            )
        except Exception as exc:
            raise RuntimeError(f"full trajectory collision check failed: {exc}") from exc
        metrics["collision_report"] = collision_report
        metrics["trajectory_waypoints"] = int(actions.shape[0])
        if not bool(collision_report.get("available", False)):
            return {
                "ok": False,
                "stop_reason": "collision_check_unavailable",
                "metrics": metrics,
            }
        if bool(collision_report.get("colliding", False)):
            return {
                "ok": False,
                "stop_reason": "trajectory_collision",
                "metrics": metrics,
            }
        return {
            "ok": True,
            "actions": actions,
            "metrics": metrics,
        }

    def plan_base_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        standoff_m: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        started = time.monotonic()
        try:
            robot = self._find_robot()
            current = self._base_xy_yaw(robot)
            candidates = self._ranked_base_candidates(
                robot,
                hand=hand,
                target_xyz=target_xyz,
                standoff_m=float(standoff_m),
            )
            if not candidates:
                return {
                    "ok": False,
                    "stop_reason": "navigation_unreachable",
                    "metrics": {"candidate_count": 0, "reason": "no traversable reachable station"},
                }
            candidate_trace = [
                {
                    "xyyaw": item["xyyaw"].tolist(),
                    "geodesic_distance_m": item.get("geodesic_distance_m"),
                    "reachability_reason": item.get("reachability_reason"),
                    "reachability_stage": item.get("reachability_stage"),
                }
                for item in candidates[:8]
            ]
            base_plan_attempts = []
            base_plan = None
            best = None
            for rank, candidate in enumerate(candidates):
                remaining_s = float(timeout_s) - (time.monotonic() - started)
                if remaining_s <= 0:
                    break
                attempt = self._compute_base_plan(
                    target_xyyaw=candidate["xyyaw"],
                    timeout_s=remaining_s,
                )
                base_plan_attempts.append(
                    {
                        "rank": rank,
                        "xyyaw": candidate["xyyaw"].tolist(),
                        "ok": bool(attempt.get("ok")),
                        "stop_reason": attempt.get("stop_reason"),
                        "metrics": attempt.get("metrics", {}),
                    }
                )
                if attempt.get("ok"):
                    best = candidate["xyyaw"]
                    base_plan = attempt
                    break
            if base_plan is None or best is None:
                timed_out = time.monotonic() - started >= float(timeout_s)
                return {
                    "ok": False,
                    "stop_reason": "timeout" if timed_out else "base_plan_failed",
                    "metrics": {
                        "candidate_count": len(candidates),
                        "candidate_trace": candidate_trace,
                        "base_plan_attempts": base_plan_attempts,
                        "current_base": current.tolist(),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    },
                }
            metrics = {
                **base_plan.get("metrics", {}),
                "candidate_count": len(candidates),
                "candidate_trace": candidate_trace,
                "base_plan_attempts": base_plan_attempts,
                "post_base_reachability_required": True,
                "base_goal": best.tolist(),
                "current_base": current.tolist(),
            }
            return {
                "ok": True,
                "joint_trajectory": base_plan["joint_trajectory"],
                "base_goal": best.tolist(),
                "metrics": metrics,
            }
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": "planner_unavailable",
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                },
            }

    def _ranked_base_candidates(
        self,
        robot: Any,
        *,
        hand: str,
        target_xyz: np.ndarray,
        standoff_m: float,
    ) -> list[dict[str, Any]]:
        scene = self._scene(robot)
        trav_map = getattr(scene, "trav_map", None)
        if trav_map is None:
            raise RuntimeError("BEHAVIOR scene has no traversability map")
        current = self._base_xy_yaw(robot)
        floor = self._current_floor(scene, current)
        ranked = []
        for candidate in _base_candidates(target_xyz, standoff_m=standoff_m):
            if not self._candidate_is_traversable(trav_map, candidate, floor=floor):
                continue
            path, distance = trav_map.get_shortest_path(
                floor,
                current[:2],
                candidate[:2],
                entire_path=True,
                robot=robot,
            )
            if path is None or distance is None:
                continue
            reachable, reason, reach_metrics = self.check_arm_reachability(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=None,
                base_xyyaw=candidate,
            )
            if not reachable:
                continue
            ranked.append(
                {
                    "xyyaw": candidate,
                    "geodesic_distance_m": float(distance),
                    "reachability_reason": reason,
                    "reachability": reach_metrics,
                    "reachability_stage": reach_metrics.get("reachability_stage"),
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item["geodesic_distance_m"]),
                abs(_wrap_angle(item["xyyaw"][2] - current[2])),
            )
        )
        return ranked

    def _scene(self, robot: Any) -> Any:
        scene = getattr(robot, "scene", None)
        if scene is not None:
            return scene
        candidates = [
            self.env_facade,
            getattr(self.env_facade, "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_direct_process", None),
            getattr(getattr(getattr(self.env_facade, "_env", None), "_direct_process", None), "env", None),
        ]
        for candidate in candidates:
            scene = getattr(candidate, "scene", None)
            if scene is not None:
                return scene
        raise RuntimeError("could not locate BEHAVIOR scene for traversability map")

    def _current_floor(self, scene: Any, current_xyyaw: np.ndarray) -> int:
        floor_heights = getattr(getattr(scene, "trav_map", None), "floor_heights", None)
        if floor_heights is None:
            return 0
        heights = np.asarray(_jsonable(floor_heights), dtype=np.float64).reshape(-1)
        if heights.size == 0:
            return 0
        z = float(current_xyyaw[3]) if current_xyyaw.shape[0] > 3 else 0.0
        return int(np.argmin(np.abs(heights - z)))

    def _candidate_is_traversable(
        self,
        trav_map: Any,
        candidate: np.ndarray,
        *,
        floor: int,
    ) -> bool:
        try:
            map_xy = np.asarray(trav_map.world_to_map(candidate[:2]), dtype=np.int64).reshape(2)
            floor_maps = getattr(trav_map, "floor_map", None)
            if floor_maps is None or floor < 0 or floor >= len(floor_maps):
                return False
            trav = floor_maps[floor]
            array = np.asarray(_jsonable(trav))
            row, column = int(map_xy[0]), int(map_xy[1])
            if (
                array.ndim != 2
                or row < 0
                or column < 0
                or row >= array.shape[0]
                or column >= array.shape[1]
            ):
                return False
            return bool(array[row, column])
        except Exception:
            return False

    def _compute_base_plan(self, *, target_xyyaw: np.ndarray, timeout_s: float) -> dict[str, Any]:
        generator = self._generator(kind="base")
        self._active_generator = generator
        emb_sel = self._embodiment_cls.BASE
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        base = self._base_xy_yaw(self._find_robot())
        pos = np.array([target_xyyaw[0], target_xyyaw[1], base[3]], dtype=np.float64)
        quat = _yaw_to_quat_xyzw(float(target_xyyaw[2]))
        batch_size = max(int(generator.batch_size), 1)
        planner_targets = torch.as_tensor(pos, dtype=torch.float32).reshape(1, 3).repeat(
            batch_size,
            1,
        )
        planner_quats = torch.as_tensor(quat, dtype=torch.float32).reshape(1, 4).repeat(
            batch_size,
            1,
        )
        successes, paths = generator.compute_trajectories(
            planner_targets,
            planner_quats,
            max_attempts=5,
            timeout=min(float(timeout_s), 8.0),
            ik_fail_return=5,
            enable_finetune_trajopt=True,
            finetune_attempts=2,
            return_full_result=False,
            success_ratio=1.0 / batch_size,
            ik_only=False,
            skip_obstacle_update=False,
            emb_sel=emb_sel,
        )
        success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
        success_indices = np.flatnonzero(success_array)
        metrics = {
            "successes": success_array.tolist(),
            "ik_only": False,
            "curobo_config": str(self._base_config_path()),
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": batch_size,
        }
        if success_indices.size == 0:
            return {"ok": False, "stop_reason": "base_plan_failed", "metrics": metrics}
        path = paths[int(success_indices[0])]
        q_traj = generator.path_to_joint_trajectory(
            path,
            get_full_js=True,
            emb_sel=emb_sel,
        )
        q_traj = _interpolate_joint_trajectory(q_traj, max_inter_dist=0.01)
        collision_report = self._check_q_trajectory_collisions(generator, q_traj)
        metrics.update(
            {
                "trajectory_waypoints": int(len(q_traj)),
                "collision_report": collision_report,
            }
        )
        if not bool(collision_report.get("available", False)):
            return {
                "ok": False,
                "stop_reason": "collision_check_unavailable",
                "metrics": metrics,
            }
        if bool(collision_report.get("colliding", False)):
            return {"ok": False, "stop_reason": "trajectory_collision", "metrics": metrics}
        return {
            "ok": True,
            "joint_trajectory": np.asarray(_jsonable(q_traj), dtype=np.float32),
            "metrics": metrics,
        }

    def _initial_joint_pos_for_base_candidate(self, robot: Any, base_xyyaw: np.ndarray) -> np.ndarray:
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64).reshape(-1).copy()
        base_idx = _indices(getattr(robot, "base_idx", []))
        if len(base_idx) >= 6:
            q[base_idx[0]] = float(base_xyyaw[0])
            q[base_idx[1]] = float(base_xyyaw[1])
            q[base_idx[5]] = float(base_xyyaw[2])
            return q
        control_idx = _indices(getattr(robot, "base_control_idx", []))
        if len(control_idx) >= 3:
            q[control_idx[0]] = float(base_xyyaw[0])
            q[control_idx[1]] = float(base_xyyaw[1])
            q[control_idx[2]] = float(base_xyyaw[2])
            return q
        raise RuntimeError("R1Pro base DOF indices unavailable for candidate IK")

    def _base_xy_yaw(self, robot: Any) -> np.ndarray:
        try:
            link = robot.links[getattr(robot, "base_footprint_link_name", "base_link")]
            pos, quat = link.get_position_orientation()
            pos = np.asarray(_jsonable(pos), dtype=np.float64)
            quat = np.asarray(_jsonable(quat), dtype=np.float64)
            yaw = _yaw_from_quat_xyzw(quat)
            return np.array([pos[0], pos[1], yaw, pos[2]], dtype=np.float64)
        except Exception:
            qpos = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
            idx = _indices(getattr(robot, "base_control_idx", [0, 1, 5]))
            values = qpos[idx[:3]]
            return np.asarray([values[0], values[1], values[2], 0.0], dtype=np.float64)

    def get_base_pose(self) -> np.ndarray:
        return self._base_xy_yaw(self._find_robot())[:3]

    def _linear_base_actions(self, start: np.ndarray, goal: np.ndarray, *, timeout_s: float) -> np.ndarray:
        steps = max(1, min(int(timeout_s * 20), 240))
        actions = np.zeros((steps, ACTION_DIM), dtype=np.float32)
        delta = (goal - start) / steps
        actions[:, ENV_ACTION_SEGMENTS["base"]] = delta.reshape(1, 3)
        return validate_action_chunk(actions)

    def q_trajectory_to_actions(self, q_traj: Any, *, hand: str | None) -> np.ndarray:
        robot = self._find_robot()
        q = np.asarray(_jsonable(q_traj), dtype=np.float64)
        if q.ndim != 2:
            raise RuntimeError(f"cuRobo q trajectory must be [T,D], got {q.shape}")
        hold = self.hold_action()
        actions = []
        q_names = list(getattr(robot, "joints", {}).keys())
        for row in q:
            action = self.joint_target_to_action(row, hand=hand)
            actions.append(self._apply_latches_and_inactive_segments(action, hold, hand=hand))

        try:
            if q_names:
                names_path = self.output_dir / "planner_curobo_configs" / "last_q_joint_names.json"
                names_path.parent.mkdir(parents=True, exist_ok=True)
                names_path.write_text(json.dumps(q_names, indent=2), encoding="utf-8")
        except Exception:
            pass
        return validate_action_chunk(np.stack(actions, axis=0))

    def joint_target_to_action(self, q: Any, *, hand: str | None) -> np.ndarray:
        """Mirror R1Pro q_to_action while preserving smooth 1D gripper commands."""
        robot = self._find_robot()
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        target = torch.as_tensor(np.asarray(_jsonable(q)), dtype=torch.float32)
        expected_layout = (
            ("base", 3),
            ("trunk", 4),
            ("arm_left", 7),
            ("gripper_left", 1),
            ("arm_right", 7),
            ("gripper_right", 1),
        )
        controllers = list(getattr(robot, "controllers", {}).items())
        actual_layout = tuple(
            (str(name), int(controller.command_dim))
            for name, controller in controllers
        )
        if actual_layout != expected_layout:
            raise RuntimeError(
                "R1Pro controller layout does not match the 23D env action contract: "
                f"expected={expected_layout!r} actual={actual_layout!r}"
            )

        action_parts = []
        for name, controller in controllers:
            if name.startswith("gripper_"):
                side = name.removeprefix("gripper_")
                action_parts.append(
                    torch.as_tensor([self._gripper_latch(side)], dtype=torch.float32)
                )
                continue
            command = target[_indices(controller.dof_idx)]
            if name == "base":
                command = self._base_target_to_local_command(robot, command)
            action_parts.append(controller._reverse_preprocess_command(command))

        action = np.asarray(
            _jsonable(torch.cat(action_parts, dim=0)),
            dtype=np.float32,
        ).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(
                f"R1Pro controller packer returned {action.shape}, expected [{ACTION_DIM}]"
            )
        hold = self.hold_action() if hand is not None else action
        return self._apply_latches_and_inactive_segments(action, hold, hand=hand)

    def _base_target_to_local_command(self, robot: Any, command: Any) -> Any:
        import torch
        from omnigibson.utils import transform_utils as transform_utils
        from omnigibson.utils.geometry_utils import wrap_angle

        current_rz = robot.get_joint_positions()[_indices(robot.base_idx)[5]]
        delta_yaw = wrap_angle(command[2] - current_rz)
        body_position, body_orientation = robot.get_position_orientation()
        canonical_position = torch.as_tensor(
            [command[0], command[1], body_position[2]],
            dtype=torch.float32,
        )
        local_position = transform_utils.relative_pose_transform(
            canonical_position,
            torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            body_position,
            body_orientation,
        )[0]
        return torch.stack([local_position[0], local_position[1], delta_yaw])

    def hold_action(self, hand: str | None = None) -> np.ndarray:
        del hand
        robot = self._find_robot()
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
        action = self.joint_target_to_action(q, hand=None)
        for side in ("left", "right"):
            action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = self._gripper_latch(side)
        return validate_action_chunk(action.reshape(1, ACTION_DIM))[0]

    def _apply_latches_and_inactive_segments(
        self,
        action: np.ndarray,
        hold: np.ndarray,
        *,
        hand: str | None,
    ) -> np.ndarray:
        _verify_env_action_segments()
        out = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        hold = np.asarray(hold, dtype=np.float32).reshape(ACTION_DIM)
        if hand is None:
            for segment in ("trunk", "left_arm", "right_arm"):
                out[ENV_ACTION_SEGMENTS[segment]] = hold[ENV_ACTION_SEGMENTS[segment]]
        else:
            hand = _normalize_hand(hand)
            out[ENV_ACTION_SEGMENTS["base"]] = hold[ENV_ACTION_SEGMENTS["base"]]
            inactive = "right" if hand == "left" else "left"
            out[ENV_ACTION_SEGMENTS[f"{inactive}_arm"]] = hold[ENV_ACTION_SEGMENTS[f"{inactive}_arm"]]
        for side in ("left", "right"):
            out[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = self._gripper_latch(side)
        return out

    def _gripper_latch(self, hand: str) -> float:
        value = getattr(self.env_facade, "_gripper_latch", {}).get(hand, 1.0)
        return float(value)

    def joint_margin(self) -> float | None:
        report = self.joint_margin_report()
        return report.get("min_normalized_margin") if report.get("available") else None

    def joint_margin_report(self) -> dict[str, Any]:
        robot = self._find_robot()
        try:
            q = np.asarray(_jsonable(robot.get_joint_positions(normalized=True)), dtype=np.float64)
            controlled = _indices(getattr(robot, "trunk_control_idx", []))
            arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
            for side in ("left", "right"):
                controlled.extend(_indices(arm_control_idx.get(side, [])))
            controlled = sorted(set(controlled))
            if len(controlled) != 18:
                raise RuntimeError(
                    "expected 18 trunk+arm controlled joints for R1Pro, "
                    f"got {len(controlled)}"
                )
            relevant = q[controlled]
            if not np.isfinite(relevant).all():
                raise RuntimeError("normalized trunk/arm joint state is non-finite")
            margin = float(np.min(1.0 - np.abs(relevant)))
            return {
                "available": True,
                "min_normalized_margin": margin,
                "threshold_normalized": 0.03,
                "threshold_raw_rad": 0.05,
                "ok": bool(margin >= 0.03),
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_normalized_margin": None,
                "threshold_normalized": 0.03,
                "threshold_raw_rad": 0.05,
                "ok": None,
            }

    def _check_q_trajectory_collisions(
        self,
        generator: Any,
        q_traj: Any,
        *,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        if not hasattr(generator, "check_collisions"):
            report = {"available": False, "reason": "check_collisions_unavailable", "min_margin_m": None}
            self._last_collision_report = report
            return report
        try:
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]
            q_tensor = torch.as_tensor(
                np.asarray(_jsonable(q_traj), dtype=np.float32),
                dtype=torch.float32,
            )
            collision_chunks = []
            waypoint_count = int(q_tensor.shape[0])
            for start in range(0, waypoint_count, 16):
                collision_chunks.append(
                    generator.check_collisions(
                        q_tensor[start : start + 16],
                        self_collision_check=True,
                        skip_obstacle_update=start > 0,
                        attached_obj=attached_obj,
                    )
                )
            colliding = np.concatenate(
                [np.asarray(_jsonable(chunk), dtype=bool).reshape(-1) for chunk in collision_chunks]
            )
        except Exception as exc:
            report = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_margin_m": None,
            }
            self._last_collision_report = report
            return report
        array = np.asarray(_jsonable(colliding), dtype=bool)
        report = {
            "available": True,
            "colliding": bool(array.any()),
            "collision_waypoints": int(array.sum()),
            "checked_waypoints": int(array.size),
            "min_margin_m": None,
            "margin_available": False,
            "attached_collision_body": {"available": attached_obj is not None},
        }
        self._last_collision_report = report
        self._last_collision_step = int(getattr(self.env_facade, "_env_steps", -1))
        return report

    def collision_report(self) -> dict[str, Any]:
        step = int(getattr(self.env_facade, "_env_steps", -1))
        if (
            self._last_collision_report.get("available")
            and step >= 0
            and self._last_collision_step >= 0
            and step - self._last_collision_step < self._collision_check_interval_steps
        ):
            return dict(self._last_collision_report)
        try:
            generator = self._active_generator or self._generator(kind="arm", hand="left")
            robot = self._find_robot()
            q = robot.get_joint_positions().reshape(1, -1)
            attached: dict[str, Any] = {}
            for side in ("left", "right"):
                item = self.get_attached_object(side)
                if item:
                    attached.update(item)
            self._check_q_trajectory_collisions(
                generator,
                q,
                attached_obj=attached or None,
            )
        except Exception as exc:
            self._last_collision_report = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_margin_m": None,
            }
            self._last_collision_step = step
        return dict(self._last_collision_report)

    def collision_margin(self) -> float | None:
        return self._last_collision_report.get("min_margin_m")

    def contact_report(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray | None = None,
        allowed_contact_distance_m: float = 0.025,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        robot = self._find_robot()
        finder = getattr(robot, "_find_gripper_contacts", None)
        if finder is None:
            return {
                "available": False,
                "reason": "gripper_contact_api_unavailable",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        try:
            contacts, _contact_links = finder(arm=hand, return_contact_positions=True)
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        points = []
        for item in contacts:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    points.append(np.asarray(_jsonable(item[1]), dtype=np.float64).reshape(3))
                except Exception:
                    pass
        min_distance = None
        if points and target_xyz is not None:
            target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
            min_distance = float(min(np.linalg.norm(point - target) for point in points))
        expected = bool(points and min_distance is not None and min_distance <= float(allowed_contact_distance_m))
        unexpected = bool(points and not expected)
        return {
            "available": True,
            "contact_count": int(len(points)),
            "min_contact_target_distance_m": min_distance,
            "allowed_contact_distance_m": float(allowed_contact_distance_m),
            "unexpected_contact": unexpected,
            "expected_contact": expected,
        }

    def get_attached_object(self, hand: str) -> Any:
        hand = _normalize_hand(hand)
        robot = self._find_robot()
        obj = None
        try:
            obj = getattr(robot, "_ag_obj_in_hand", {}).get(hand)
        except Exception:
            obj = None
        if obj is not None:
            self._attached_objects_by_hand[hand] = obj
        obj = self._attached_objects_by_hand.get(hand)
        if obj is None:
            return None
        return {EEF_LINK_BY_HAND[hand]: obj}

    def clear_attached_object(self, hand: str) -> None:
        self._attached_objects_by_hand.pop(_normalize_hand(hand), None)


def _indices(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(x) for x in np.asarray(_jsonable(value), dtype=np.int64).reshape(-1)]
    except Exception:
        return []


def _contains_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return False


def _verify_env_action_segments() -> None:
    covered = []
    for segment in ENV_ACTION_SEGMENTS.values():
        covered.extend(range(segment.start, segment.stop))
    if covered != list(range(ACTION_DIM)):
        raise RuntimeError("ENV_ACTION_SEGMENTS no longer covers the 23D env action exactly")


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat_xyzw(quat: Any) -> float:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    half = float(yaw) * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def _base_candidates(target_xyz: np.ndarray, *, standoff_m: float) -> list[np.ndarray]:
    candidates = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        xy = target_xyz[:2] - standoff_m * np.array([math.cos(angle), math.sin(angle)])
        yaw = math.atan2(target_xyz[1] - xy[1], target_xyz[0] - xy[0])
        candidates.append(np.array([xy[0], xy[1], yaw], dtype=np.float64))
    return candidates


class PlannerExecutor:
    """Executes planner tool requests inside the BEHAVIOR env process."""

    def __init__(
        self,
        *,
        env: Any,
        frame_cache: FrameCache,
        output_dir: str | Path | None = None,
        backend: Any | None = None,
        max_stall_steps: int = 20,
    ) -> None:
        self.env = env
        self.frame_cache = frame_cache
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self.backend = backend if backend is not None else RealCuroboBackend(env, output_dir=self.output_dir)
        self.max_stall_steps = int(max_stall_steps)
        self.last_info: Any = None

    def observe(self, camera: str) -> dict[str, Any]:
        return self.frame_cache.observe_payload(canonical_camera(camera))

    def pixel_to_world(
        self,
        *,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        try:
            if u is None or v is None:
                raise CameraGeometryError("both u=column and v=row are required")
            frame = self.frame_cache.get_current(canonical_camera(camera), str(frame_id))
            projection = backproject_pixel_to_world(
                frame,
                u=u,
                v=v,
                depth_window_px=int(depth_window_px),
                output_frame=output_frame,
            )
            metrics = {
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "step_index": frame.step_index,
                "confidence": projection["confidence"],
                "reprojection_error_px": projection["reprojection_error_px"],
                "depth": projection["depth"],
            }
            return primitive_result(
                primitive_success=True,
                task_success=False,
                stop_reason="projected",
                recoverable=True,
                suggested_next_tool="move_to",
                metrics=metrics,
                diagnostics={
                    "xyz": projection["xyz"],
                    "surface_normal": projection["surface_normal"],
                    "output_frame": output_frame,
                },
            )
        except Exception as exc:
            return primitive_result(
                primitive_success=False,
                task_success=False,
                stop_reason="projection_failed",
                recoverable=True,
                suggested_next_tool="observe",
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )

    def navigate_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        standoff_m: float = 0.85,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            hand = _normalize_hand(hand)
            target = self._world_target(target_xyz, frame=frame)
            plan = self.backend.plan_base_trajectory(
                hand=hand,
                target_xyz=target,
                standoff_m=float(standoff_m),
                timeout_s=float(timeout_s),
            )
            if not plan.get("ok"):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=str(plan.get("stop_reason", "base_plan_failed")),
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics=plan.get("metrics", {}),
                    diagnostics=plan,
                )
            actions = plan.get("actions")
            execution = self._execute_actions(
                validate_action_chunk(actions) if actions is not None else None,
                hand=hand,
                target_xyz=None,
                target_quat_xyzw=None,
                position_tolerance_m=0.05,
                orientation_tolerance_rad=math.radians(5.0),
                timeout_s=float(timeout_s),
                require_pose=False,
                base_goal_xyyaw=np.asarray(plan.get("base_goal"), dtype=np.float64),
                joint_trajectory=plan.get("joint_trajectory"),
            )
            reachable, reason, reach_metrics = self.backend.check_arm_reachability(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=None,
            )
            metrics = {**plan.get("metrics", {}), **execution["metrics"], **reach_metrics}
            metrics["elapsed_s"] = round(time.monotonic() - started, 3)
            metrics["post_base_reachability_stage"] = reach_metrics.get("reachability_stage")
            success = bool(execution["primitive_success"] and reachable)
            return primitive_result(
                primitive_success=success,
                task_success=self._task_success(),
                stop_reason="arrived" if success else reason,
                recoverable=True,
                suggested_next_tool="observe",
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    def move_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        target_quat_xyzw: Any | None = None,
        plan_only: bool = False,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.087,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            hand = _normalize_hand(hand)
            target = self._world_target(target_xyz, frame=frame)
            quat = _quat_xyzw(target_quat_xyzw)
            reachable, reason, reach_metrics = self.backend.check_arm_reachability(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
            )
            if not reachable:
                suggested = "navigate_to" if reason == "navigation_required" else None
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=reason,
                    recoverable=True,
                    suggested_next_tool=suggested,
                    metrics=reach_metrics,
                )
            plan = self.backend.plan_arm_trajectory(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=float(timeout_s),
                attached_obj=_call_optional_arg(self.backend, "get_attached_object", hand),
            )
            if not plan.get("ok"):
                stop_reason = str(plan.get("stop_reason", "arm_plan_failed"))
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=stop_reason in {"unreachable", "planner_unavailable"},
                    suggested_next_tool="navigate_to" if stop_reason == "unreachable" else None,
                    metrics={**reach_metrics, **plan.get("metrics", {})},
                    diagnostics=plan,
                )
            actions = validate_action_chunk(plan["actions"])
            if plan_only:
                return primitive_result(
                    primitive_success=True,
                    task_success=self._task_success(),
                    stop_reason="plan_ready",
                    recoverable=True,
                    metrics={
                        **reach_metrics,
                        **plan.get("metrics", {}),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    },
                )
            execution = self._execute_actions(
                actions,
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                position_tolerance_m=float(position_tolerance_m),
                orientation_tolerance_rad=float(orientation_tolerance_rad),
                timeout_s=float(timeout_s),
                require_pose=True,
            )
            metrics = {**reach_metrics, **plan.get("metrics", {}), **execution["metrics"]}
            metrics["elapsed_s"] = round(time.monotonic() - started, 3)
            return primitive_result(
                primitive_success=execution["primitive_success"],
                task_success=self._task_success(),
                stop_reason=execution["stop_reason"],
                recoverable=execution["recoverable"],
                suggested_next_tool=execution["suggested_next_tool"],
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="navigate_to")

    def pick(
        self,
        *,
        hand: str,
        target_xyz: Any,
        approach_vector: Any | None = None,
        grasp_quat_xyzw: Any | None = None,
        pregrasp_offset_m: float = 0.08,
        lift_m: float = 0.08,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        try:
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            approach = _approach_vector(approach_vector)
            pregrasp = target - approach * float(pregrasp_offset_m)
            move = self.move_to(
                hand=hand,
                target_xyz=pregrasp,
                target_quat_xyzw=grasp_quat_xyzw,
                timeout_s=min(float(timeout_s), 45.0),
            )
            if not move["primitive_success"]:
                move["suggested_next_tool"] = move.get("suggested_next_tool") or "move_to"
                return move
            guarded = self._guarded_incremental_move(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=grasp_quat_xyzw,
                direction=approach,
                allow_expected_contact=True,
                position_tolerance_m=0.015,
                timeout_s=min(float(timeout_s), 30.0),
            )
            if not guarded["primitive_success"]:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=guarded["stop_reason"],
                    recoverable=guarded["recoverable"],
                    suggested_next_tool="observe",
                    metrics=guarded["metrics"],
                    diagnostics=guarded["diagnostics"],
                )
            close = self._gripper_command(hand, opening=0.0, timeout_s=5.0)
            if not close["primitive_success"]:
                return close
            attached_obj = _call_optional_arg(self.backend, "get_attached_object", hand)
            lift_target = target + np.array([0.0, 0.0, float(lift_m)], dtype=np.float64)
            lift = self.move_to(hand=hand, target_xyz=lift_target, timeout_s=min(float(timeout_s), 30.0))
            lift["stop_reason"] = "picked" if lift["primitive_success"] else lift["stop_reason"]
            lift["metrics"]["attached_collision_body"] = {"available": attached_obj is not None}
            return lift
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    def rotate_wrist(
        self,
        *,
        hand: str,
        target_quat_xyzw: Any | None = None,
        relative_axis_angle: Any | None = None,
        frame: str = "world",
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        try:
            hand = _normalize_hand(hand)
            if (target_quat_xyzw is None) == (relative_axis_angle is None):
                raise ValueError("provide exactly one of target_quat_xyzw or relative_axis_angle")
            current = self.backend.get_eef_pose(hand)
            if current is None:
                raise RuntimeError("cannot read current EEF pose for rotate_wrist")
            position, current_quat = current
            if target_quat_xyzw is not None:
                target_quat = _quat_xyzw(target_quat_xyzw)
            else:
                rel = _axis_angle_to_quat_xyzw(relative_axis_angle)
                if frame == "eef":
                    target_quat = _quat_multiply_xyzw(current_quat, rel)
                elif frame == "world":
                    target_quat = _quat_multiply_xyzw(rel, current_quat)
                else:
                    raise ValueError("frame must be 'world' or 'eef'")
            return self.move_to(
                hand=hand,
                target_xyz=position,
                target_quat_xyzw=target_quat,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="move_to")

    def press(
        self,
        *,
        hand: str,
        target_xyz: Any,
        press_direction: Any | None = None,
        approach_distance_m: float = 0.04,
        press_depth_m: float = 0.012,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        try:
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            direction = _approach_vector(press_direction if press_direction is not None else [0, 0, -1])
            pre = target - direction * float(approach_distance_m)
            move = self.move_to(hand=hand, target_xyz=pre, timeout_s=min(float(timeout_s), 30.0))
            if not move["primitive_success"]:
                return move
            contact = target + direction * float(press_depth_m)
            guarded_press = self._guarded_incremental_move(
                hand=hand,
                target_xyz=contact,
                target_quat_xyzw=None,
                direction=direction,
                allow_expected_contact=True,
                position_tolerance_m=0.012,
                timeout_s=min(float(timeout_s), 20.0),
            )
            return primitive_result(
                primitive_success=guarded_press["primitive_success"],
                task_success=self._task_success(),
                stop_reason="pressed" if guarded_press["primitive_success"] else guarded_press["stop_reason"],
                recoverable=guarded_press["recoverable"],
                suggested_next_tool=None if guarded_press["primitive_success"] else "observe",
                metrics=guarded_press["metrics"],
                diagnostics=guarded_press["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    def release(
        self,
        *,
        hand: str,
        opening: float = 1.0,
        retreat_vector: Any | None = None,
        retreat_m: float = 0.03,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        try:
            hand = _normalize_hand(hand)
            release = self._gripper_command(hand, opening=float(opening), timeout_s=min(float(timeout_s), 5.0))
            if not release["primitive_success"] or retreat_vector is None or float(retreat_m) <= 0:
                return release
            current = self.backend.get_eef_pose(hand)
            if current is None:
                return release
            direction = _approach_vector(retreat_vector)
            target = current[0] + direction * float(retreat_m)
            retreat = self.move_to(hand=hand, target_xyz=target, timeout_s=min(float(timeout_s), 20.0))
            retreat["stop_reason"] = "released" if retreat["primitive_success"] else retreat["stop_reason"]
            return retreat
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="move_to")

    def _guarded_incremental_move(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: Any | None,
        direction: np.ndarray,
        allow_expected_contact: bool,
        position_tolerance_m: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        started = time.monotonic()
        quat = _quat_xyzw(target_quat_xyzw)
        current = self.backend.get_eef_pose(hand)
        if current is None:
            return self._execution_result(
                primitive_success=False,
                stop_reason="pose_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
            )
        start = np.asarray(current[0], dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        total = float(np.linalg.norm(target - start))
        steps = max(1, int(math.ceil(total / 0.002)))
        if steps > max(1, int(float(timeout_s) * 120)):
            return self._execution_result(
                primitive_success=False,
                stop_reason="timeout",
                recoverable=True,
                suggested_next_tool="move_to",
                executed=0,
                trace=[],
                final_pos_err=total,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"guarded_step_m": 0.002, "guarded_waypoints": steps},
            )
        trace: list[dict[str, Any]] = []
        executed = 0
        final_pos_err = total
        final_ori_err: float | None = None
        for index in range(1, steps + 1):
            if time.monotonic() - started > float(timeout_s):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={"guarded_step_m": 0.002, "guarded_waypoints": steps},
                )
            waypoint = start + (target - start) * (index / steps)
            contact = self._contact_report(
                hand=hand,
                target_xyz=target,
                allowed_contact_distance_m=max(0.025, float(position_tolerance_m) * 2.0),
            )
            if contact.get("available") is False:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="contact_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="observe",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "guarded_step_m": 0.002,
                        "guarded_waypoints": steps,
                        "contact_report": contact,
                    },
                )
            if self._contact_is_abort(
                contact,
                hand=hand,
                target_xyz=target,
                allow_expected_contact=allow_expected_contact,
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="unexpected_contact",
                    recoverable=True,
                    suggested_next_tool="observe",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "guarded_step_m": 0.002,
                        "guarded_waypoints": steps,
                        "contact_report": contact,
                    },
                )
            plan = self.backend.plan_arm_trajectory(
                hand=hand,
                target_xyz=waypoint,
                target_quat_xyzw=quat,
                timeout_s=min(2.0, max(0.25, float(timeout_s) - (time.monotonic() - started))),
                attached_obj=_call_optional_arg(self.backend, "get_attached_object", hand),
            )
            if not plan.get("ok"):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=str(plan.get("stop_reason", "guarded_plan_failed")),
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={"guarded_step_m": 0.002, "guarded_waypoints": steps, **plan.get("metrics", {})},
                )
            hold_required = 10 if index == steps else 1
            execution = self._execute_actions(
                validate_action_chunk(plan["actions"]),
                hand=hand,
                target_xyz=waypoint,
                target_quat_xyzw=quat,
                position_tolerance_m=max(0.003, min(float(position_tolerance_m), 0.006)),
                orientation_tolerance_rad=0.087,
                timeout_s=min(3.0, max(0.5, float(timeout_s) - (time.monotonic() - started))),
                require_pose=True,
                hold_steps_required=hold_required,
                contact_target_xyz=target,
                allow_expected_contact=allow_expected_contact,
            )
            executed += int(execution["metrics"].get("executed_waypoints", 0))
            trace.extend(execution["diagnostics"].get("trace", []))
            final_pos_err = execution["metrics"].get("final_position_error_m")
            final_ori_err = execution["metrics"].get("final_orientation_error_rad")
            if not execution["primitive_success"]:
                execution["metrics"].update({"guarded_step_m": 0.002, "guarded_waypoints": steps})
                return execution
        return self._execution_result(
            primitive_success=True,
            stop_reason="guarded_reached",
            recoverable=True,
            suggested_next_tool=None,
            executed=executed,
            trace=trace,
            final_pos_err=final_pos_err,
            final_ori_err=final_ori_err,
            held_steps=10,
            started=started,
            extra_metrics={
                "guarded_step_m": 0.002,
                "guarded_waypoints": steps,
                "guarded_direction": direction.tolist(),
            },
        )

    def _gripper_command(self, hand: str, *, opening: float, timeout_s: float) -> dict[str, Any]:
        command = 1.0 if float(opening) >= 0.5 else -1.0
        latch = getattr(self.env, "_gripper_latch", None)
        if isinstance(latch, dict):
            latch[hand] = command
        hold = _call_optional_arg(self.backend, "hold_action", hand)
        if hold is None:
            hold = np.zeros((ACTION_DIM,), dtype=np.float32)
        actions = np.repeat(
            np.asarray(hold, dtype=np.float32).reshape(1, ACTION_DIM),
            max(1, int(timeout_s * 10)),
            axis=0,
        )
        segment = ENV_ACTION_SEGMENTS[f"{hand}_gripper"]
        actions[:, segment] = command
        execution = self._execute_actions(
            validate_action_chunk(actions),
            hand=hand,
            target_xyz=None,
            target_quat_xyzw=None,
            position_tolerance_m=0.0,
            orientation_tolerance_rad=0.0,
            timeout_s=timeout_s,
            require_pose=False,
        )
        if command > 0 and execution["primitive_success"]:
            clear_attached = getattr(self.backend, "clear_attached_object", None)
            if callable(clear_attached):
                clear_attached(hand)
        return primitive_result(
            primitive_success=execution["primitive_success"],
            task_success=self._task_success(),
            stop_reason="gripper_commanded" if execution["primitive_success"] else execution["stop_reason"],
            recoverable=execution["recoverable"],
            suggested_next_tool=None,
            metrics=execution["metrics"],
            diagnostics=execution["diagnostics"],
        )

    def _execute_actions(
        self,
        actions: np.ndarray | None,
        *,
        hand: str,
        target_xyz: np.ndarray | None,
        target_quat_xyzw: np.ndarray | None,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        timeout_s: float,
        require_pose: bool,
        base_goal_xyyaw: np.ndarray | None = None,
        hold_steps_required: int = 10,
        contact_target_xyz: np.ndarray | None = None,
        allow_expected_contact: bool = False,
        joint_trajectory: Any | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if (actions is None) == (joint_trajectory is None):
            raise ValueError("provide exactly one of actions or joint_trajectory")
        action_chunk = validate_action_chunk(actions) if actions is not None else None
        q_chunk = (
            np.asarray(_jsonable(joint_trajectory), dtype=np.float32)
            if joint_trajectory is not None
            else None
        )
        if q_chunk is not None and (q_chunk.ndim != 2 or q_chunk.shape[0] < 1):
            raise ValueError(f"joint trajectory must be [T,D], got {q_chunk.shape}")
        planned_steps = action_chunk.shape[0] if action_chunk is not None else q_chunk.shape[0]
        dynamics = (
            _action_dynamics_report(action_chunk)
            if action_chunk is not None
            else {
                "ok": True,
                "mode": "online_controller_commands",
                "max_velocity_command_delta": 0.0,
                "max_acceleration_command_delta": 0.0,
                "velocity_limit": 5.0,
                "acceleration_limit": 10.0,
            }
        )
        if not dynamics["ok"]:
            return self._execution_result(
                primitive_success=False,
                stop_reason=str(dynamics["stop_reason"]),
                recoverable=True,
                suggested_next_tool="move_to",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"dynamics": dynamics},
            )
        deadline = time.monotonic() + float(timeout_s)
        best_error = float("inf")
        last_error = float("inf")
        stalled_steps = 0
        held_steps = 0
        executed = 0
        trace: list[dict[str, Any]] = []
        final_pos_err: float | None = None
        final_ori_err: float | None = None
        hold_action: np.ndarray | None = None
        previous_action: np.ndarray | None = None
        previous_delta: np.ndarray | None = None
        index = 0
        max_steps = planned_steps + int(hold_steps_required) + self.max_stall_steps
        while executed < max_steps:
            if time.monotonic() > deadline:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                )
            if index < planned_steps:
                if action_chunk is not None:
                    action = action_chunk[index]
                else:
                    action = self.backend.joint_target_to_action(
                        q_chunk[index],
                        hand=None,
                    )
                index += 1
            else:
                if hold_action is None:
                    hold_action = _call_optional_arg(self.backend, "hold_action", hand)
                    if hold_action is None:
                        hold_action = np.zeros((ACTION_DIM,), dtype=np.float32)
                action = hold_action
            action = validate_action_chunk(
                np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
            )[0]
            if previous_action is not None:
                delta = action - previous_action
                max_velocity = float(np.max(np.abs(delta)))
                dynamics["max_velocity_command_delta"] = max(
                    float(dynamics["max_velocity_command_delta"]),
                    max_velocity,
                )
                if previous_delta is not None:
                    max_acceleration = float(np.max(np.abs(delta - previous_delta)))
                    dynamics["max_acceleration_command_delta"] = max(
                        float(dynamics["max_acceleration_command_delta"]),
                        max_acceleration,
                    )
                    if max_acceleration > float(dynamics["acceleration_limit"]):
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="acceleration_limit",
                            recoverable=True,
                            suggested_next_tool="move_to",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={"dynamics": dynamics},
                        )
                previous_delta = delta
                if max_velocity > float(dynamics["velocity_limit"]):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="velocity_limit",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"dynamics": dynamics},
                    )
            previous_action = action.copy()
            joint_report = self._joint_margin_report()
            if not bool(joint_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            if joint_report.get("ok") is False:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_margin",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            collision_report = self._collision_report()
            if not bool(collision_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="collision_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"collision_report": collision_report},
                )
            collision_margin = collision_report.get("min_margin_m")
            if bool(collision_report.get("colliding", False)) or (
                collision_margin is not None and float(collision_margin) < 0.0
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="unexpected_collision",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"collision_report": collision_report},
                )
            self._step_env_action(action)
            executed += 1
            if base_goal_xyyaw is not None:
                base_pose = _call_optional(self.backend, "get_base_pose")
                if base_pose is None:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="pose_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
                base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
                final_pos_err = float(np.linalg.norm(base_pose[:2] - base_goal_xyyaw[:2]))
                final_ori_err = abs(_wrap_angle(float(base_pose[2]) - float(base_goal_xyyaw[2])))
            else:
                pose = self.backend.get_eef_pose(hand) if hasattr(self.backend, "get_eef_pose") else None
                if pose is None:
                    if require_pose:
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="pose_feedback_unavailable",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                        )
                    final_pos_err = None
                    final_ori_err = None
                else:
                    pos, quat = pose
                    if target_xyz is not None:
                        final_pos_err = float(np.linalg.norm(pos - target_xyz))
                    if target_quat_xyzw is not None:
                        final_ori_err = _quat_angle_error_rad(quat, target_quat_xyzw)
            if contact_target_xyz is not None:
                contact = self._contact_report(
                    hand=hand,
                    target_xyz=contact_target_xyz,
                    allowed_contact_distance_m=max(0.025, float(position_tolerance_m) * 2.0),
                )
                if not bool(contact.get("available", False)):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="contact_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"contact_report": contact},
                    )
                if self._contact_is_abort(
                    contact,
                    hand=hand,
                    target_xyz=contact_target_xyz,
                    allow_expected_contact=allow_expected_contact,
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="unexpected_contact",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"contact_report": contact},
                    )
            trace.append(
                {
                    "step": executed,
                    "position_error_m": final_pos_err,
                    "orientation_error_rad": final_ori_err,
                    "joint_margin": joint_report,
                    "collision_report": collision_report,
                    "hold_step": index >= planned_steps,
                }
            )
            position_ok = final_pos_err is None or final_pos_err <= position_tolerance_m
            orientation_ok = final_ori_err is None or final_ori_err <= orientation_tolerance_rad
            if position_ok and orientation_ok:
                held_steps += 1
                if held_steps >= int(hold_steps_required):
                    return self._execution_result(
                        primitive_success=True,
                        stop_reason="reached",
                        recoverable=True,
                        suggested_next_tool=None,
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"dynamics": dynamics},
                    )
                continue
            held_steps = 0
            error = final_pos_err if final_pos_err is not None else 0.0
            if final_ori_err is not None:
                error += final_ori_err
            if error + 1e-5 < best_error:
                best_error = error
                stalled_steps = 0
            elif abs(last_error - error) < 1e-5:
                stalled_steps += 1
            else:
                stalled_steps = 0
            last_error = error
            if stalled_steps >= self.max_stall_steps:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="stalled_tracking",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"dynamics": dynamics},
                )
            if index >= planned_steps and stalled_steps >= max(1, self.max_stall_steps // 2):
                break
        return self._execution_result(
            primitive_success=False,
            stop_reason="target_tolerance_not_met",
            recoverable=True,
            suggested_next_tool="move_to",
            executed=executed,
            trace=trace,
            final_pos_err=final_pos_err,
            final_ori_err=final_ori_err,
            held_steps=held_steps,
            started=started,
            extra_metrics={"dynamics": dynamics},
        )

    def _execution_result(
        self,
        *,
        primitive_success: bool,
        stop_reason: str,
        recoverable: bool,
        suggested_next_tool: str | None,
        executed: int,
        trace: list[dict[str, Any]],
        final_pos_err: float | None,
        final_ori_err: float | None,
        held_steps: int = 0,
        started: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        joint_report = self._joint_margin_report()
        collision_report = self._collision_report()
        metrics = {
            "executed_waypoints": int(executed),
            "final_position_error_m": final_pos_err,
            "final_orientation_error_rad": final_ori_err,
            "held_steps": int(held_steps),
            "elapsed_s": round(time.monotonic() - started, 3) if started is not None else None,
            "joint_margin": joint_report,
            "collision_report": collision_report,
            "collision_margin": collision_report.get("min_margin_m"),
        }
        metrics.update(extra_metrics or {})
        return {
            "primitive_success": bool(primitive_success),
            "stop_reason": stop_reason,
            "recoverable": bool(recoverable),
            "suggested_next_tool": suggested_next_tool,
            "metrics": metrics,
            "diagnostics": {"trace": trace[-50:]},
        }

    def _joint_margin_report(self) -> dict[str, Any]:
        report = _call_optional(self.backend, "joint_margin_report")
        if isinstance(report, dict):
            return report
        margin = _call_optional(self.backend, "joint_margin")
        if margin is None:
            return {
                "available": False,
                "reason": "joint_margin_unavailable",
                "min_normalized_margin": None,
                "threshold_normalized": 0.03,
                "threshold_raw_rad": 0.05,
                "ok": None,
            }
        return {
            "available": True,
            "min_normalized_margin": float(margin),
            "threshold_normalized": 0.03,
            "threshold_raw_rad": 0.05,
            "ok": bool(float(margin) >= 0.03),
        }

    def _collision_report(self) -> dict[str, Any]:
        report = _call_optional(self.backend, "collision_report")
        if isinstance(report, dict):
            return report
        margin = _call_optional(self.backend, "collision_margin")
        return {
            "available": margin is not None,
            "reason": None if margin is not None else "collision_margin_unavailable",
            "colliding": bool(margin is not None and float(margin) < 0.0),
            "min_margin_m": float(margin) if margin is not None else None,
            "margin_available": margin is not None,
        }

    def _contact_report(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray | None,
        allowed_contact_distance_m: float,
    ) -> dict[str, Any]:
        report = _call_optional_kw(
            self.backend,
            "contact_report",
            hand=hand,
            target_xyz=target_xyz,
            allowed_contact_distance_m=allowed_contact_distance_m,
        )
        if isinstance(report, dict):
            return report
        return {
            "available": False,
            "reason": "contact_report_unavailable",
            "unexpected_contact": False,
            "expected_contact": False,
        }

    def _contact_is_abort(
        self,
        contact: dict[str, Any],
        *,
        hand: str,
        target_xyz: np.ndarray,
        allow_expected_contact: bool,
    ) -> bool:
        if bool(contact.get("unexpected_contact", False)):
            return True
        if not bool(contact.get("expected_contact", False)):
            return False
        if not allow_expected_contact:
            return True
        current = self.backend.get_eef_pose(hand) if hasattr(self.backend, "get_eef_pose") else None
        if current is None:
            return True
        distance = float(np.linalg.norm(np.asarray(current[0], dtype=np.float64) - target_xyz))
        allowed = contact.get("allowed_contact_distance_m", 0.025)
        try:
            allowed = float(allowed)
        except Exception:
            allowed = 0.025
        return distance > max(allowed, 0.025)

    def _step_env_action(self, action: np.ndarray) -> None:
        ret = self.env.chunk_step(np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM))
        if isinstance(ret, tuple) and len(ret) >= 5:
            self.last_info = ret[4]

    def _world_target(self, target_xyz: Any, *, frame: str) -> np.ndarray:
        if str(frame) != "world":
            raise ValueError("planner currently requires frame='world'")
        return _as_xyz(target_xyz)

    def _task_success(self) -> bool:
        return official_task_success(self.last_info or getattr(self.env, "_last_info", None))

    def _exception_result(self, exc: Exception, *, suggested_next_tool: str | None) -> dict[str, Any]:
        return primitive_result(
            primitive_success=False,
            task_success=self._task_success(),
            stop_reason="error",
            recoverable=True,
            suggested_next_tool=suggested_next_tool,
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        )


def _call_optional(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _call_optional_arg(obj: Any, name: str, *args: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:
        return None


def _call_optional_kw(obj: Any, name: str, **kwargs: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(**kwargs)
    except Exception:
        return None


def _action_dynamics_report(actions: np.ndarray) -> dict[str, Any]:
    chunk = validate_action_chunk(actions)
    if chunk.shape[0] < 2:
        return {
            "ok": True,
            "max_velocity_command_delta": 0.0,
            "max_acceleration_command_delta": 0.0,
            "velocity_limit": 5.0,
            "acceleration_limit": 10.0,
        }
    velocity = np.diff(chunk, axis=0)
    acceleration = np.diff(velocity, axis=0) if velocity.shape[0] >= 2 else np.zeros((0, ACTION_DIM))
    max_velocity = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    max_acceleration = float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
    velocity_limit = 5.0
    acceleration_limit = 10.0
    if max_velocity > velocity_limit:
        return {
            "ok": False,
            "stop_reason": "velocity_limit",
            "max_velocity_command_delta": max_velocity,
            "max_acceleration_command_delta": max_acceleration,
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
        }
    if max_acceleration > acceleration_limit:
        return {
            "ok": False,
            "stop_reason": "acceleration_limit",
            "max_velocity_command_delta": max_velocity,
            "max_acceleration_command_delta": max_acceleration,
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
        }
    return {
        "ok": True,
        "max_velocity_command_delta": max_velocity,
        "max_acceleration_command_delta": max_acceleration,
        "velocity_limit": velocity_limit,
        "acceleration_limit": acceleration_limit,
    }


def _interpolate_joint_trajectory(
    trajectory: Any,
    *,
    max_inter_dist: float,
) -> np.ndarray:
    q = np.asarray(_jsonable(trajectory), dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 1:
        raise ValueError(f"joint trajectory must be [T,D], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("joint trajectory contains NaN or infinity")
    if max_inter_dist <= 0:
        raise ValueError("max_inter_dist must be positive")
    if q.shape[0] == 1:
        return q.copy()

    interpolated = []
    for start, end in zip(q[:-1], q[1:], strict=True):
        intervals = max(
            1,
            int(math.ceil(float(np.max(np.abs(end - start))) / max_inter_dist)),
        )
        for index in range(intervals):
            alpha = index / intervals
            interpolated.append(start + (end - start) * alpha)
    interpolated.append(q[-1])
    result = np.stack(interpolated, axis=0).astype(np.float32, copy=False)
    if result.shape[0] > 1:
        max_delta = float(np.max(np.abs(np.diff(result, axis=0))))
        if max_delta > max_inter_dist + 1e-6:
            raise RuntimeError(
                f"interpolated joint delta {max_delta} exceeds {max_inter_dist}"
            )
    return result


def _approach_vector(value: Any | None) -> np.ndarray:
    if value is None:
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        vec = _as_xyz(value, name="approach_vector")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError("approach/press vector cannot be zero")
    return vec / norm


__all__ = [
    "LEFT_EEF_LINK",
    "RIGHT_EEF_LINK",
    "PlannerExecutor",
    "RealCuroboBackend",
    "official_task_success",
    "primitive_result",
]
