#!/usr/bin/env python3
"""Real-radio planner bootstrap and visual acceptance harness.

This harness never injects surrogate objects.  It uses the task's real
``radio_receiver.n.01_1`` only.  Simulator truth is confined to the private
bootstrap/audit report; planner tool inputs and results keep the RGB-D-only
public contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.behavior.env_server import BehaviorEnvFacade, _load_env_config
from robots.behavior.snapshot_manifest import (
    build_snapshot_manifest,
    write_snapshot_manifest,
)

RADIO_BDDL_NAME = "radio_receiver.n.01_1"
CAMERAS = ("head", "left_wrist", "right_wrist")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--activity-instance-id", type=int, default=211)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--standoff-m", type=float, default=0.85)
    parser.add_argument("--navigation-timeout-s", type=float, default=180.0)
    return parser.parse_args()


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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _quaternion_error_rad(first: Any, second: Any) -> float:
    first_array = np.asarray(_jsonable(first), dtype=np.float64).reshape(4)
    second_array = np.asarray(_jsonable(second), dtype=np.float64).reshape(4)
    first_array /= np.linalg.norm(first_array)
    second_array /= np.linalg.norm(second_array)
    cosine = float(np.clip(abs(np.dot(first_array, second_array)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def _pose_error(reference: tuple[Any, Any], current: tuple[Any, Any]) -> dict[str, float]:
    return {
        "position_m": float(
            np.linalg.norm(
                np.asarray(_jsonable(reference[0]), dtype=np.float64)
                - np.asarray(_jsonable(current[0]), dtype=np.float64)
            )
        ),
        "orientation_rad": _quaternion_error_rad(reference[1], current[1]),
    }


def _attachment_audit(env: BehaviorEnvFacade) -> dict[str, Any]:
    robot = env._planner.backend._find_robot()
    assisted = getattr(robot, "_ag_obj_in_hand", {})
    return {
        hand: {
            "backend": env._planner.backend.get_attached_object(hand) is not None,
            "simulator": assisted.get(hand) is not None,
        }
        for hand in ("left", "right")
    }


def _assert_unattached(attachment_audit: dict[str, Any]) -> None:
    if any(
        bool(source_attached)
        for hand in attachment_audit.values()
        for source_attached in hand.values()
    ):
        raise RuntimeError("radio-near state unexpectedly holds an object")


def _radio_mask(
    sensor: Any, *, radio_object_name: str
) -> tuple[np.ndarray, dict[int, str]]:
    if sensor is None or "seg_instance_id" not in sensor.modalities:
        raise RuntimeError("radio audit requires private instance segmentation")
    private_obs, private_info = sensor.get_obs()
    labels = {
        int(key): str(value)
        for key, value in private_info["seg_instance_id"].items()
    }
    object_path_segment = f"/{str(radio_object_name).strip().lower()}/"
    radio_ids = {
        key
        for key, value in labels.items()
        if object_path_segment in f"/{value.strip('/').lower()}/"
    }
    segmentation = private_obs["seg_instance_id"]
    try:
        import torch

        if torch.is_tensor(segmentation):
            segmentation = segmentation.detach().cpu().numpy()
    except Exception:
        pass
    image = np.asarray(segmentation).squeeze()
    if not radio_ids:
        # A wrist camera can legitimately have no radio label because the
        # object is outside that camera's frustum.  Preserve a same-sized
        # empty mask; the head-camera visibility gate is enforced by caller.
        return np.zeros_like(image, dtype=bool), {}
    mask = np.isin(image, sorted(radio_ids))
    return mask, {key: labels[key] for key in sorted(radio_ids)}


def _save_camera_audit(
    env: BehaviorEnvFacade,
    output_dir: Path,
    *,
    radio_object_name: str,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for camera in CAMERAS:
        frame = env._frame_cache.latest(camera)
        sensor = env._sensor_for_camera(camera)
        mask, labels = _radio_mask(
            sensor, radio_object_name=radio_object_name
        )
        if mask.shape != frame.rgb.shape[:2]:
            raise RuntimeError(
                f"{camera} private mask {mask.shape} does not match RGB "
                f"{frame.rgb.shape[:2]}"
            )
        rgb_path = output_dir / f"{camera}.png"
        mask_path = output_dir / f"{camera}_radio_mask.png"
        imageio.imwrite(rgb_path, np.asarray(frame.rgb, dtype=np.uint8)[..., :3])
        imageio.imwrite(mask_path, (mask.astype(np.uint8) * 255))
        rows, cols = np.nonzero(mask)
        result[camera] = {
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "radio_visible": bool(rows.size),
            "radio_pixels": int(rows.size),
            "radio_bbox_uv": (
                [
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max()),
                    int(rows.max()),
                ]
                if rows.size
                else None
            ),
            "private_instance_labels": labels,
            "rgb_path": str(rgb_path),
            "rgb_sha256": _sha256(rgb_path),
            "mask_path": str(mask_path),
            "mask_sha256": _sha256(mask_path),
        }
    return result


def main() -> None:
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "radio_visual_acceptance.json"
    instance_path = (
        Path(args.activity_instance_dir)
        / (
            "house_double_floor_lower_task_turning_on_radio_0_"
            f"{args.activity_instance_id}_template-tro_state.json"
        )
    )
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
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "process": {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ports": [],
        },
        "commit": _git(repo, "rev-parse", "HEAD"),
        "worktree_dirty": bool(_git(repo, "status", "--porcelain")),
        "configuration": vars(env_args),
        "instance": {
            "path": str(instance_path),
            "sha256": _sha256(instance_path),
        },
        "surrogate_fixtures_used": False,
        "private_bootstrap_truth_used": True,
        "navigation_timeout_s": float(args.navigation_timeout_s),
        "bootstrap_tool_calls": [],
    }

    def save() -> None:
        report_path.write_text(
            json.dumps(_jsonable(report), indent=2), encoding="utf-8"
        )

    save()
    env: BehaviorEnvFacade | None = None
    try:
        cfg = _load_env_config(env_args)
        env = BehaviorEnvFacade(
            cfg=cfg,
            meta=vars(env_args),
            output_dir=output_dir,
            control_mode="planner_tools",
        )
        _observation, reset_info = env.reset()
        og_env = env._env.omnigibson_env
        radio = og_env.task.object_scope.get(RADIO_BDDL_NAME)
        if radio is None or not radio.exists:
            raise RuntimeError(f"real task object is missing: {RADIO_BDDL_NAME}")
        radio_object = getattr(radio, "wrapped_obj", radio)
        radio_object_name = str(getattr(radio_object, "name", ""))
        if not radio_object_name:
            raise RuntimeError("real radio has no stable simulator object name")
        from omnigibson.object_states.toggle import ToggledOn

        radio_position, radio_orientation = radio.get_position_orientation()
        toggled_before = bool(radio.states[ToggledOn].get_value())
        if toggled_before:
            raise RuntimeError("radio must be off in the clean bootstrap state")
        report["reset"] = {
            "official_success": bool(
                isinstance(reset_info, dict)
                and isinstance(reset_info.get("done"), dict)
                and reset_info["done"].get("success", False)
            ),
            "radio_toggled_on": toggled_before,
        }
        report["audit_private"] = {
            "purpose": "bootstrap_and_visual_identity_only_not_public_tool_payload",
            "radio_bddl_name": RADIO_BDDL_NAME,
            "radio_simulator_name": radio_object_name,
            "radio_position_world": _jsonable(radio_position),
            "radio_orientation_xyzw": _jsonable(radio_orientation),
        }

        navigation_dir = output_dir / "navigation_to_real_radio"
        env.start_video_segment(navigation_dir / "episode.mp4")
        started = time.monotonic()
        navigation = env.navigate_to(
            hand="right",
            target_xyz=_jsonable(radio_position),
            frame="world",
            standoff_m=float(args.standoff_m),
            timeout_s=float(args.navigation_timeout_s),
        )
        navigation_elapsed = round(time.monotonic() - started, 3)
        report["bootstrap_tool_calls"].append(
            {
                "tool": "navigate_to",
                "target_source": "private_bootstrap_real_radio_identity",
                "result": navigation,
                "wall_elapsed_s": navigation_elapsed,
            }
        )
        if navigation.get("primitive_success") is not True:
            raise RuntimeError(
                "real-radio navigate_to failed: "
                f"{navigation.get('stop_reason', 'unknown')}"
            )

        # Force one post-navigation synchronized capture for both visual audit
        # and the next public pixel_to_world call.  No object pose is placed in
        # the public observation or planner tool result.
        env._refresh_observation_without_step()
        assert env._last_observation is not None
        env._append_video(env._last_observation)
        visibility = _save_camera_audit(
            env,
            navigation_dir / "post_rgb",
            radio_object_name=radio_object_name,
        )
        if not visibility["head"]["radio_visible"]:
            raise RuntimeError("head camera cannot see the real radio after navigation")

        attachment_before_snapshot = _attachment_audit(env)
        _assert_unattached(attachment_before_snapshot)
        if bool(radio.states[ToggledOn].get_value()):
            raise RuntimeError("navigation unexpectedly toggled the radio")

        import torch

        restore_video_dir = output_dir / "snapshot_restore"
        env.start_video_segment(restore_video_dir / "episode.mp4")
        radio_pose_before_snapshot = radio.get_position_orientation()
        robot = env._planner.backend._find_robot()
        robot_pose_before_snapshot = robot.get_position_orientation()
        robot_joints_before_snapshot = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        )
        stale_frame_id = env._frame_cache.latest("head").frame_id
        snapshot = env.dump_simulator_state(serialized=True).detach().cpu()
        snapshot_path = output_dir / "radio_near_state.pt"
        torch.save(snapshot, snapshot_path)
        report["radio_near_snapshot"] = {
            "path": str(snapshot_path),
            "sha256": _sha256(snapshot_path),
            "serialized_elements": int(snapshot.numel()),
            "source": "real_curobo_navigate_to_with_private_bootstrap_target",
            "radio_visible_after_restore_required": True,
            "radio_toggled_on": False,
            "attached_objects": {"left": False, "right": False},
        }

        # Immediately prove that the exact artifact is loadable and retains
        # the visible, untoggled, unattached real-radio setup.
        env.restore_simulator_state(snapshot, serialized=True)
        restored_visibility = _save_camera_audit(
            env,
            output_dir / "snapshot_restore_rgb",
            radio_object_name=radio_object_name,
        )
        if not restored_visibility["head"]["radio_visible"]:
            raise RuntimeError("restored state does not preserve head radio visibility")
        if bool(radio.states[ToggledOn].get_value()):
            raise RuntimeError("restored radio-near state is not clean/off")
        attachment_after_restore = _attachment_audit(env)
        _assert_unattached(attachment_after_restore)
        radio_restore_error = _pose_error(
            radio_pose_before_snapshot, radio.get_position_orientation()
        )
        robot_restore_error = _pose_error(
            robot_pose_before_snapshot, robot.get_position_orientation()
        )
        robot_joints_after_restore = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        )
        joint_restore_error = float(
            np.max(np.abs(robot_joints_before_snapshot - robot_joints_after_restore))
        )
        radio_navigation_error = _pose_error(
            (radio_position, radio_orientation), radio.get_position_orientation()
        )
        if radio_restore_error["position_m"] > 1e-4:
            raise RuntimeError("restored radio position differs from saved snapshot")
        if radio_restore_error["orientation_rad"] > 1e-3:
            raise RuntimeError("restored radio orientation differs from saved snapshot")
        if robot_restore_error["position_m"] > 1e-4:
            raise RuntimeError("restored robot position differs from saved snapshot")
        if robot_restore_error["orientation_rad"] > 1e-3:
            raise RuntimeError("restored robot orientation differs from saved snapshot")
        if joint_restore_error > 1e-3:
            raise RuntimeError("restored robot joints differ from saved snapshot")
        stale_frame_rejected = False
        try:
            env._frame_cache.get_current("head", stale_frame_id)
        except Exception:
            stale_frame_rejected = True
        if not stale_frame_rejected:
            raise RuntimeError("pre-restore RGB-D frame id remained valid")
        invariants = {
            "radio_bddl_name": RADIO_BDDL_NAME,
            "radio_simulator_name": radio_object_name,
            "head_radio_visible": True,
            "radio_toggled_on": False,
            "attachments": attachment_after_restore,
            "radio_restore_error": radio_restore_error,
            "radio_navigation_error": radio_navigation_error,
            "robot_restore_error": robot_restore_error,
            "max_joint_restore_error_rad": joint_restore_error,
            "pre_restore_frame_rejected": stale_frame_rejected,
        }
        try:
            import omnigibson as og

            omnigibson_version = getattr(og, "__version__", None)
        except Exception:
            omnigibson_version = None
        status_porcelain = _git(repo, "status", "--porcelain")
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
        )
        manifest = build_snapshot_manifest(
            snapshot_path,
            serialized_elements=int(snapshot.numel()),
            serialized_dtype=str(snapshot.dtype),
            serialized_shape=list(snapshot.shape),
            serialized_finite=bool(torch.isfinite(snapshot).all().item()),
            meta=vars(env_args),
            source={
                "commit": report["commit"],
                "worktree_dirty": report["worktree_dirty"],
                "status_porcelain_sha256": hashlib.sha256(
                    status_porcelain.encode("utf-8")
                ).hexdigest(),
                "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
                "generator": str(Path(__file__).resolve()),
            },
            omnigibson_version=(
                None if omnigibson_version is None else str(omnigibson_version)
            ),
            invariants=invariants,
        )
        snapshot_manifest = write_snapshot_manifest(snapshot_path, manifest)
        report["radio_near_snapshot"]["manifest_path"] = str(snapshot_manifest)
        report["radio_near_snapshot"]["manifest_sha256"] = _sha256(
            snapshot_manifest
        )
        report["radio_near_snapshot"]["invariants"] = invariants
        report["visual_audit"] = {
            "after_navigation": visibility,
            "after_immediate_restore": restored_visibility,
        }
        report["status"] = "passed"
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        if env is not None:
            env.close()
        report["video_artifacts"] = {}
        for name, path in {
            "initial_reset": output_dir / "episode.mp4",
            "navigation_to_real_radio": (
                output_dir / "navigation_to_real_radio" / "episode.mp4"
            ),
            "snapshot_restore": output_dir / "snapshot_restore" / "episode.mp4",
        }.items():
            meta_path = path.parent / "video_meta.json"
            artifact: dict[str, Any] = {
                "path": str(path),
                "exists": path.is_file(),
            }
            if path.is_file():
                artifact.update(
                    {
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            if meta_path.is_file():
                artifact["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            report["video_artifacts"][name] = artifact
        save()


if __name__ == "__main__":
    main()
