import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior.serial_eval import (
    TURNING_ON_RADIO_PUBLIC_IDS,
    EvalEntry,
    _checkout_identity,
    _file_fingerprint,
    _gpu_lock_path,
    _green_center_marker_visible,
    _owned_group_members,
    _run_entry,
    _terminal_press_wrist_image,
    _unverified_group_members,
    _validate_entry_python,
    build_entry_argv,
    read_turning_on_radio_instances,
    select_instances,
    validate_instance_result,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_png(path: Path, *, center_color=(0, 190, 0)) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), color=(210, 10, 10))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 88, 88), fill=(205, 200, 190))
    draw.ellipse((18, 18, 78, 78), fill=(24, 22, 18))
    draw.ellipse((40, 40, 56, 56), fill=center_color)
    image.save(path, format="PNG")


def _entry(tmp_path, instance_id=242, seed=0):
    state_path = (
        tmp_path / "instances" / f"radio_0_{instance_id}_template-tro_state.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")
    return EvalEntry(
        split_position=0,
        csv_position=0,
        activity_instance_id=instance_id,
        seed=seed,
        output_dir=tmp_path / "run",
        argv=("python", "-m", "rpent.cli.main"),
        checkpoint=tmp_path / "checkpoint",
        cuda_device="7",
        instance_state_path=state_path,
        instance_state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
    )


def _write_bound_artifacts(entry, *, success, commit="abc", status="stopped"):
    _write_json(
        entry.output_dir / "run_manifest.json",
        {
            "commit": commit,
            "worktree_dirty": False,
            "control_mode": "pi0_nav_pick_vla",
            "stage3_press_enabled": True,
            "status": status,
            "task": {
                "suite": "behavior_2025_challenge",
                "task": 0,
                "task_name": "turning_on_radio",
                "activity_definition_id": 0,
                "activity_instance_id": entry.activity_instance_id,
                "activity_instance_dir": str(
                    entry.instance_state_path.parent.resolve()
                ),
                "scene_model": "house_double_floor_lower",
                "seed": entry.seed,
                "max_episode_steps": 24756,
            },
            "checkpoint": str(entry.checkpoint.resolve()),
            "gpu": entry.cuda_device,
            "processes": {
                "env": {
                    "managed": True,
                    "pid": None,
                    "stopped_at": "done",
                },
                "vla": {
                    "managed": True,
                    "pid": None,
                    "stopped_at": "done",
                },
            },
        },
    )
    _write_json(
        entry.output_dir / "final_result.json",
        {
            "run_status": "completed",
            "task_success": success,
            "official_success_source": 'info["done"]["success"]',
            "runtime_cleanup": "complete",
            "error": None,
        },
    )
    if success:
        image = entry.output_dir / "live" / "press.png"
        _write_png(image)
        checkpoint_root = entry.output_dir / "state_checkpoints"
        checkpoint1 = checkpoint_root / "state_checkpoint_1.json"
        run_binding = {
            "nonce": "same-run-test",
            "suite": "behavior_2025_challenge",
            "task": 0,
            "task_name": "turning_on_radio",
            "activity_definition_id": 0,
            "activity_instance_id": entry.activity_instance_id,
            "scene_model": "house_double_floor_lower",
            "seed": entry.seed,
        }
        _write_json(
            checkpoint1,
            {
                "schema_version": 1,
                "kind": "robot_motion_checkpoint",
                "not_simulator_restore": True,
                "checkpoint_name": "state_checkpoint_1",
                "stage": "post_pi0_nav_pick",
                "held_hand": "left",
                "press_hand": "right",
                "object_name": "radio_test",
                "run_binding": run_binding,
            },
        )
        projection_id = "button_projection_test"
        checkpoint2 = checkpoint_root / "state_checkpoint_2.json"
        _write_json(
            checkpoint2,
            {
                "schema_version": 1,
                "kind": "robot_motion_checkpoint",
                "not_simulator_restore": True,
                "checkpoint_name": "state_checkpoint_2",
                "stage": "pre_press_alignment",
                "env_step": 100,
                "held_hand": "left",
                "press_hand": "right",
                "object_name": "radio_test",
                "run_binding": run_binding,
                "prepress": {
                    "source_checkpoint_path": str(checkpoint1),
                    "source_checkpoint_sha256": hashlib.sha256(
                        checkpoint1.read_bytes()
                    ).hexdigest(),
                    "button_gate": {
                        "button_visible": True,
                        "face_class": "BUTTON_FACE",
                        "positive_signature_complete": True,
                        "camera": "press_wrist",
                        "resolved_camera": "right_wrist",
                        "frame_id": "capture:100:right_wrist",
                        "capture_group_id": "capture:100",
                        "env_step": 100,
                        "gate_id": "button_gate_test",
                    },
                    "button_projection": {
                        "projection_id": projection_id,
                        "gate_id": "button_gate_test",
                        "camera": "press_wrist",
                        "resolved_camera": "right_wrist",
                        "frame_id": "capture:100:right_wrist",
                        "capture_group_id": "capture:100",
                        "env_step": 100,
                        "projection_metrics": {
                            "camera": "right_wrist",
                            "frame_id": "capture:100:right_wrist",
                            "step_index": 100,
                        },
                    },
                },
            },
        )
        frame_id = "capture:104:right_wrist"
        capture_group = {"id": "capture:104"}
        metadata = entry.output_dir / "live" / "press.json"
        _write_json(
            metadata,
            {
                "camera": "right_wrist",
                "frame_id": frame_id,
                "capture_group": capture_group,
                "total_env_steps": 104,
                "rgb_path": str(image),
                "metadata_path": str(metadata),
            },
        )
        trace = [
            {
                "step": 8,
                "tool": "save_robot_state_checkpoint",
                "result": {
                    "state_checkpoint_2_path": str(checkpoint2),
                    "state_checkpoint_2_sha256": hashlib.sha256(
                        checkpoint2.read_bytes()
                    ).hexdigest(),
                    "held_hand": "left",
                    "press_hand": "right",
                },
            },
            {
                "step": 9,
                "tool": "post_pick_direct_finger_toggle",
                "result": {
                    "task_success": True,
                    "press_hand": "right",
                    "projection_id": projection_id,
                },
            },
            {
                "step": 10,
                "tool": "post_success_hold_frames",
                "input": {"frames": 4},
                "result": {
                    "primitive_success": True,
                    "task_success": True,
                    "requested_frames": 4,
                    "executed_frames": 4,
                    "start_env_step": 100,
                    "end_env_step": 104,
                },
            },
            {
                "step": 11,
                "tool": "observe",
                "input": {"camera": "press_wrist"},
                "result": {
                    "task_success": True,
                    "camera": "press_wrist",
                    "resolved_camera": "right_wrist",
                    "frame_id": frame_id,
                    "capture_group": capture_group,
                    "total_env_steps": 104,
                    "visual_review": {
                        "rgb_path": str(image),
                        "metadata_path": str(metadata),
                    },
                },
            },
        ]
        trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in trace),
            encoding="utf-8",
        )


def test_csv_protocol_preserves_authoritative_order(tmp_path):
    csv_path = tmp_path / "test_instances.csv"
    csv_path.write_text(
        "Task ID,Task,Public Test Instance IDs\n"
        '0,turning_on_radio,"'
        + ", ".join(str(value) for value in TURNING_ON_RADIO_PUBLIC_IDS)
        + '"\n',
        encoding="utf-8",
    )
    expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    public = read_turning_on_radio_instances(csv_path, expected_sha256=expected_hash)

    assert public == TURNING_ON_RADIO_PUBLIC_IDS
    assert select_instances(public, "official_first10") == public[:10]
    assert select_instances(public, "holdback_last10") == public[10:]
    assert select_instances(public, "all_public") == public


def test_entry_argv_owns_fresh_stage3_runtime_and_has_no_resume_flags(tmp_path):
    argv = build_entry_argv(
        python=Path("/python"),
        repo_root=tmp_path.resolve(),
        output_dir=tmp_path / "run",
        behavior_repo=tmp_path / "behavior",
        behavior_python=Path("/behavior-python"),
        checkpoint=tmp_path / "checkpoint",
        activity_instance_id=211,
        seed=0,
        cuda_device="7",
        model="gpt-5.6",
        max_turns=200,
        cerebrum_timeout_s=7200,
    )

    assert argv.count("--activity-instance-id") == 1
    assert argv[argv.index("--activity-instance-id") + 1] == "211"
    assert "--behavior-stage3-press" in argv
    assert "--cerebrum" in argv and argv[argv.index("--cerebrum") + 1] == "codex"
    assert "--no-driver" not in argv
    assert "--env-port" not in argv
    assert "--vla-endpoint" not in argv
    assert not any("mirror" in value.lower() for value in argv)
    assert not any("restore" in value.lower() for value in argv)


def test_result_classification_uses_raw_success_not_exit_code(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "task_failed"
    assert errors == []

    _write_bound_artifacts(entry, success=True)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=1,
        timed_out=False,
    )
    assert outcome == "run_error"
    assert "top-level RPent process returned nonzero" in errors


def test_missing_final_result_with_nonzero_exit_is_run_error(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    (entry.output_dir / "final_result.json").unlink()

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=1,
        timed_out=False,
    )

    assert outcome == "run_error"
    assert "missing or invalid final_result.json" in errors


def test_success_requires_post_hold_press_wrist_evidence(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    (entry.output_dir / "pi0_nav_pick_tool_trace.jsonl").unlink()

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "successful run lacks post-success render hold" in errors
    assert "successful run lacks fresh post-hold press-wrist image" in errors


def test_embedded_pi0_terminal_evidence_is_accepted(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    terminal_dir = entry.output_dir / "visual_review" / "terminal_success"
    image = terminal_dir / "right_wrist.png"
    _write_png(image)
    view = {
        "camera": "right_wrist",
        "path": str(image),
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "width": 96,
        "height": 96,
        "frame_id": "capture:104:right",
        "capture_group_id": "capture:104",
        "env_step": 104,
    }
    metadata_path = terminal_dir / "terminal_success_evidence.json"
    evidence = {
        "source": "pi0_nav_pick_internal_terminal_finalize",
        "complete": True,
        "task_success_before_hold": True,
        "task_success_after_hold": True,
        "hold_frames_requested": 4,
        "hold_frames_executed": 4,
        "start_env_step": 100,
        "end_env_step": 104,
        "held_hand": "left",
        "press_hand": "right",
        "role_resolution_source": "unique_terminal_attachment_evidence",
        "logical_camera": "press_wrist",
        "resolved_camera": "right_wrist",
        "capture_group_id": "capture:104",
        "terminal_press_wrist": view,
        "metadata_path": str(metadata_path),
    }
    _write_json(metadata_path, evidence)
    trace = {
        "step": 1,
        "tool": "pi0_nav_pick",
        "result": {
            "task_success": True,
            "terminal_success_evidence": evidence,
        },
    }
    (entry.output_dir / "pi0_nav_pick_tool_trace.jsonl").write_text(
        json.dumps(trace) + "\n", encoding="utf-8"
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "passed"
    assert errors == []
    assert _terminal_press_wrist_image(entry.output_dir) == str(image.resolve())


def test_terminal_image_must_decode_as_png(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    image = entry.output_dir / "live" / "press.png"
    image.write_bytes(b"png")

    assert _terminal_press_wrist_image(entry.output_dir) is None


def test_green_center_marker_requires_button_context(tmp_path):
    green = tmp_path / "green.png"
    red = tmp_path / "red.png"
    solid_green = tmp_path / "solid_green.png"
    unrelated_white_patch = tmp_path / "unrelated_white_patch.png"
    disconnected_white_patches = tmp_path / "disconnected_white_patches.png"
    white_semicircle = tmp_path / "white_semicircle.png"
    _write_png(green)
    _write_png(red, center_color=(210, 10, 10))
    from PIL import Image, ImageDraw

    Image.new("RGB", (96, 96), color=(0, 190, 0)).save(solid_green, format="PNG")
    false_image = Image.new("RGB", (96, 96), color=(205, 10, 10))
    false_draw = ImageDraw.Draw(false_image)
    false_draw.rectangle((22, 24, 70, 72), fill=(20, 20, 20))
    false_draw.ellipse((40, 40, 56, 56), fill=(0, 190, 0))
    false_draw.rectangle((72, 30, 92, 66), fill=(210, 205, 195))
    false_image.save(unrelated_white_patch, format="PNG")
    patches = Image.new("RGB", (96, 96), color=(205, 10, 10))
    patch_draw = ImageDraw.Draw(patches)
    patch_draw.rectangle((18, 18, 78, 78), fill=(20, 20, 20))
    patch_draw.ellipse((40, 40, 56, 56), fill=(0, 190, 0))
    for box in (
        (15, 15, 34, 28),
        (62, 15, 81, 28),
        (15, 68, 34, 81),
        (62, 68, 81, 81),
    ):
        patch_draw.rectangle(box, fill=(210, 205, 195))
    patches.save(disconnected_white_patches, format="PNG")
    semicircle = Image.new("RGB", (96, 96), color=(205, 10, 10))
    semicircle_draw = ImageDraw.Draw(semicircle)
    semicircle_draw.ellipse((18, 18, 78, 78), fill=(20, 20, 20))
    semicircle_draw.ellipse((40, 40, 56, 56), fill=(0, 190, 0))
    semicircle_draw.arc((12, 12, 84, 84), 0, 180, fill=(210, 205, 195), width=8)
    semicircle.save(white_semicircle, format="PNG")

    assert _green_center_marker_visible(green) is True
    assert _green_center_marker_visible(red) is False
    assert _green_center_marker_visible(solid_green) is False
    assert _green_center_marker_visible(unrelated_white_patch) is False
    assert _green_center_marker_visible(disconnected_white_patches) is False
    assert _green_center_marker_visible(white_semicircle) is False


def test_success_with_red_terminal_marker_is_incomplete(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    _write_png(
        entry.output_dir / "live" / "press.png",
        center_color=(210, 10, 10),
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert (
        "successful run terminal press-wrist image does not show green center marker"
        in errors
    )


def test_external_stage3_success_requires_same_run_checkpoint2(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    (entry.output_dir / "state_checkpoints" / "state_checkpoint_2.json").unlink()

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "successful run lacks fresh post-hold press-wrist image" in errors


def test_external_stage3_rejects_non_press_wrist_checkpoint_projection(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    checkpoint2 = entry.output_dir / "state_checkpoints" / "state_checkpoint_2.json"
    payload = json.loads(checkpoint2.read_text(encoding="utf-8"))
    payload["prepress"]["button_projection"]["camera"] = "head"
    _write_json(checkpoint2, payload)
    trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    saved = next(
        record for record in trace if record["tool"] == "save_robot_state_checkpoint"
    )
    saved["result"]["state_checkpoint_2_sha256"] = hashlib.sha256(
        checkpoint2.read_bytes()
    ).hexdigest()
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )

    assert _terminal_press_wrist_image(entry.output_dir) is None


def test_external_stage3_rejects_in_root_swapped_terminal_image(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    swapped = entry.output_dir / "live" / "unrelated_green.png"
    _write_png(swapped)
    trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observation = next(record for record in trace if record["tool"] == "observe")
    observation["result"]["visual_review"]["rgb_path"] = str(swapped)
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )

    assert _terminal_press_wrist_image(entry.output_dir) is None


def test_external_stage3_malformed_hold_step_fails_closed(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hold = next(
        record for record in trace if record["tool"] == "post_success_hold_frames"
    )
    hold["step"] = "malformed"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "successful run lacks post-success render hold" in errors
    assert "successful run lacks fresh post-hold press-wrist image" in errors


def test_temp_checkpoint_json_makes_run_incomplete(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    temporary = (
        entry.output_dir
        / "state_checkpoints"
        / "tmp_state_checkpoint_before_press.json"
    )
    temporary.parent.mkdir()
    temporary.write_text("{}", encoding="utf-8")

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "temporary checkpoint JSON was not deleted" in errors


def _write_proc_stat(proc_root, *, pid, ppid, pgid, sid, start_ticks, state="S"):
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    fields = [state, str(ppid), str(pgid), str(sid)] + ["0"] * 15 + [str(start_ticks)]
    (process_dir / "stat").write_text(
        f"{pid} (worker name) " + " ".join(fields), encoding="utf-8"
    )


def test_leaderless_descendant_is_reported_but_not_signal_authorized(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(
        proc_root,
        pid=102,
        ppid=1,
        pgid=100,
        sid=100,
        start_ticks=1100,
    )
    process = {
        "managed": True,
        "pid": 100,
        "pgid": 100,
        "sid": 100,
        "start_ticks": 1000,
    }

    assert _owned_group_members(process, proc_root=proc_root) == ()
    assert _unverified_group_members(process, proc_root=proc_root) == (102,)


def test_owned_group_requires_exact_live_session_leader(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(
        proc_root,
        pid=100,
        ppid=1,
        pgid=100,
        sid=100,
        start_ticks=1000,
    )
    _write_proc_stat(
        proc_root,
        pid=102,
        ppid=100,
        pgid=100,
        sid=100,
        start_ticks=1100,
    )
    process = {
        "managed": True,
        "pid": 100,
        "pgid": 100,
        "sid": 100,
        "start_ticks": 1000,
    }

    assert _owned_group_members(process, proc_root=proc_root) == (100, 102)


def test_owned_group_rejects_recycled_or_unbound_identity(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(
        proc_root,
        pid=100,
        ppid=1,
        pgid=100,
        sid=100,
        start_ticks=1100,
    )
    process = {
        "managed": True,
        "pid": 100,
        "pgid": 100,
        "sid": 100,
        "start_ticks": 1000,
    }

    assert _owned_group_members(process, proc_root=proc_root) == ()
    assert _unverified_group_members(process, proc_root=proc_root) == (100,)
    process["sid"] = 99
    assert _owned_group_members(process, proc_root=proc_root) == ()


def test_terminal_image_must_be_inside_output_and_not_symlink(tmp_path):
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    outside = tmp_path / "outside.png"
    _write_png(outside)
    trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observation = next(record for record in trace if record["tool"] == "observe")
    observation["result"]["visual_review"]["rgb_path"] = str(outside)
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )

    assert _terminal_press_wrist_image(entry.output_dir) is None

    link = entry.output_dir / "linked.png"
    link.symlink_to(outside)
    observation["result"]["visual_review"]["rgb_path"] = str(link)
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )
    assert _terminal_press_wrist_image(entry.output_dir) is None


def test_gpu_lock_is_global_and_device_specific():
    assert _gpu_lock_path("7").parent == Path("/tmp")
    assert _gpu_lock_path("7") == _gpu_lock_path(" 7 ")
    assert _gpu_lock_path("007") == _gpu_lock_path("7")
    assert _gpu_lock_path("7") != _gpu_lock_path("6")
    with pytest.raises(ValueError, match="decimal GPU ordinal"):
        _gpu_lock_path("GPU-deadbeef")


def test_dirty_checkout_fingerprint_tracks_content_not_only_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    tracked.write_text("first dirty value\n", encoding="utf-8")
    first = _checkout_identity(repo)
    tracked.write_text("second dirty value\n", encoding="utf-8")
    second = _checkout_identity(repo)

    assert first["status_sha256"] == second["status_sha256"]
    assert first["dirty_content_sha256"] != second["dirty_content_sha256"]


def test_entry_python_dependency_preflight_fails_before_eval(monkeypatch, tmp_path):
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'httpx'\n",
    )
    monkeypatch.setattr(
        "robots.behavior.serial_eval.subprocess.run", lambda *args, **kwargs: completed
    )

    with pytest.raises(RuntimeError, match="No module named 'httpx'"):
        _validate_entry_python(tmp_path / "python", repo_root=tmp_path)


def test_executable_fingerprint_preserves_virtualenv_symlink_path(tmp_path):
    target = tmp_path / "python-real"
    target.write_bytes(b"python")
    link = tmp_path / "venv-python"
    link.symlink_to(target)

    fingerprint = _file_fingerprint(link)

    assert fingerprint["path"] == str(link.absolute())
    assert fingerprint["resolved_path"] == str(target.resolve())


def test_timeout_launches_once_and_cleans_manifest_groups(monkeypatch, tmp_path):
    entry = _entry(tmp_path)
    calls = {"popen": 0, "manifest_cleanup": 0, "top_cleanup": 0}

    class Process:
        pid = 12345
        returncode = None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(entry.argv, timeout)

    def popen(*args, **kwargs):
        calls["popen"] += 1
        return Process()

    def cleanup_manifest(output_dir):
        assert output_dir == entry.output_dir
        calls["manifest_cleanup"] += 1
        return {}

    def cleanup_top(process):
        calls["top_cleanup"] += 1
        process.returncode = -15

    monkeypatch.setattr("robots.behavior.serial_eval.subprocess.Popen", popen)
    monkeypatch.setattr(
        "robots.behavior.serial_eval._terminate_manifest_processes",
        cleanup_manifest,
    )
    monkeypatch.setattr(
        "robots.behavior.serial_eval._terminate_top_process", cleanup_top
    )

    exit_code, timed_out = _run_entry(
        entry,
        repo_root=tmp_path,
        log_stream=io.BytesIO(),
        timeout_s=1,
    )

    assert exit_code == -15
    assert timed_out is True
    assert calls == {"popen": 1, "manifest_cleanup": 2, "top_cleanup": 1}
