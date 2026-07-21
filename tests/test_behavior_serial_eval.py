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
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        trace = [
            {
                "step": 10,
                "tool": "post_success_hold_frames",
                "input": {"frames": 4},
                "result": {"task_success": True},
            },
            {
                "step": 11,
                "tool": "observe",
                "input": {"camera": "press_wrist"},
                "result": {
                    "task_success": True,
                    "visual_review": {"rgb_path": str(image)},
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
    outside.write_bytes(b"png")
    trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
    trace = [
        {
            "step": 10,
            "tool": "post_success_hold_frames",
            "result": {"task_success": True},
        },
        {
            "step": 11,
            "tool": "observe",
            "input": {"camera": "press_wrist"},
            "result": {
                "task_success": True,
                "visual_review": {"rgb_path": str(outside)},
            },
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )

    assert _terminal_press_wrist_image(entry.output_dir) is None

    link = entry.output_dir / "linked.png"
    link.symlink_to(outside)
    trace[-1]["result"]["visual_review"]["rgb_path"] = str(link)
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
