"""Reproducible, strictly serial BEHAVIOR public-instance evaluation."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

TURNING_ON_RADIO_TASK_ID = 0
TURNING_ON_RADIO_TASK_NAME = "turning_on_radio"
TURNING_ON_RADIO_PUBLIC_IDS = (
    242,
    295,
    211,
    203,
    109,
    181,
    197,
    187,
    214,
    139,
    185,
    102,
    246,
    105,
    271,
    119,
    220,
    224,
    212,
    298,
)
TEST_INSTANCES_SHA256 = (
    "5cd78301ddc764158a20d4cf8c134afb2cb3bbf1f0611aa55aee34873b5b4d23"
)
TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."


@dataclass(frozen=True)
class EvalEntry:
    """One immutable public instance and its exact launch argv."""

    split_position: int
    csv_position: int
    activity_instance_id: int
    seed: int
    output_dir: Path
    argv: tuple[str, ...]
    checkpoint: Path
    cuda_device: str
    instance_state_path: Path
    instance_state_sha256: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _green_center_marker_visible(path: Path) -> bool:
    """Recognize the green button center inside its dark disk and white ring."""

    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    except Exception:
        return False
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False
    height, width = rgb.shape[:2]
    image_area = int(height * width)
    if image_area < 64:
        return False
    red, green, blue = (rgb[:, :, index] for index in range(3))
    green_mask = (
        (green >= 70)
        & (green >= red + 30)
        & (green >= blue + 25)
        & (green * 4 >= np.maximum(red, blue) * 5)
    )
    visited = np.zeros((height, width), dtype=bool)
    minimum_area = max(20, int(np.ceil(image_area * 0.0002)))
    maximum_area = int(np.floor(image_area * 0.08))
    candidates: list[tuple[int, float, float, int, int, int, int]] = []
    for seed_y, seed_x in np.argwhere(green_mask):
        y0, x0 = int(seed_y), int(seed_x)
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        area = 0
        x_sum = 0
        y_sum = 0
        min_x = max_x = x0
        min_y = max_y = y0
        while stack:
            y, x = stack.pop()
            area += 1
            x_sum += x
            y_sum += y
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if not visited[next_y, next_x] and green_mask[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if minimum_area <= area <= maximum_area:
            candidates.append(
                (
                    area,
                    x_sum / area,
                    y_sum / area,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                )
            )
    if not candidates:
        return False
    yy, xx = np.indices((height, width))
    dark = (rgb.max(axis=2) <= 90) & (rgb.mean(axis=2) <= 65)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)
    white = (rgb.min(axis=2) >= 110) & (channel_spread <= 90)
    for area, center_x, center_y, min_x, min_y, max_x, max_y in sorted(
        candidates, reverse=True
    ):
        box_width = max_x - min_x + 1
        box_height = max_y - min_y + 1
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        fill = area / float(box_width * box_height)
        if aspect > 1.8 or fill < 0.35:
            continue
        radius = float(np.sqrt(area / np.pi))
        distance = np.hypot(xx - center_x, yy - center_y)
        dark_annulus = (distance >= 1.20 * radius) & (distance < 2.40 * radius)
        white_annulus = (distance >= 2.50 * radius) & (distance < 4.50 * radius)
        if dark_annulus.sum() < 30 or white_annulus.sum() < 60:
            continue
        if float(dark[dark_annulus].mean()) < 0.45:
            continue
        if float(white[white_annulus].mean()) < 0.08:
            continue
        ring_mask = white & white_annulus
        ring_visited = np.zeros((height, width), dtype=bool)
        largest_ring: list[tuple[int, int]] = []
        for seed_y, seed_x in np.argwhere(ring_mask):
            y0, x0 = int(seed_y), int(seed_x)
            if ring_visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            ring_visited[y0, x0] = True
            component: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for next_y in range(max(0, y - 1), min(height, y + 2)):
                    for next_x in range(max(0, x - 1), min(width, x + 2)):
                        if (
                            not ring_visited[next_y, next_x]
                            and ring_mask[next_y, next_x]
                        ):
                            ring_visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            if len(component) > len(largest_ring):
                largest_ring = component
        if len(largest_ring) < max(40, int(white_annulus.sum() * 0.06)):
            continue
        connected_ring = np.zeros((height, width), dtype=bool)
        ring_y, ring_x = zip(*largest_ring)
        connected_ring[np.asarray(ring_y), np.asarray(ring_x)] = True
        angle = (np.arctan2(yy - center_y, xx - center_x) + 2 * np.pi) % (2 * np.pi)
        angular_coverage = []
        for angular_bin in range(72):
            bin_mask = (
                white_annulus
                & (angle >= angular_bin * 2 * np.pi / 72)
                & (angle < (angular_bin + 1) * 2 * np.pi / 72)
            )
            angular_coverage.append(
                bool(bin_mask.any()) and float(connected_ring[bin_mask].mean()) >= 0.08
            )
        doubled = angular_coverage + angular_coverage
        longest_run = 0
        current_run = 0
        for present in doubled:
            current_run = current_run + 1 if present else 0
            longest_run = max(longest_run, current_run)
        if min(longest_run, 72) < 38:
            continue
        return True
    return False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_turning_on_radio_instances(
    csv_path: Path,
    *,
    expected_sha256: str = TEST_INSTANCES_SHA256,
) -> tuple[int, ...]:
    """Read and verify the authoritative ordered public-instance row."""

    actual_sha256 = _sha256(csv_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "test_instances.csv SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    matches: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("Task ID") == str(TURNING_ON_RADIO_TASK_ID)
                and row.get("Task") == TURNING_ON_RADIO_TASK_NAME
            ):
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one turning_on_radio row in test_instances.csv"
        )
    raw_ids = matches[0].get("Public Test Instance IDs", "")
    try:
        instance_ids = tuple(int(item.strip()) for item in raw_ids.split(","))
    except ValueError as error:
        raise RuntimeError("invalid public instance ID list") from error
    if instance_ids != TURNING_ON_RADIO_PUBLIC_IDS:
        raise RuntimeError(
            "turning_on_radio public IDs differ from the pinned ordered protocol"
        )
    if len(instance_ids) != len(set(instance_ids)):
        raise RuntimeError("turning_on_radio public IDs contain duplicates")
    return instance_ids


def select_instances(instance_ids: tuple[int, ...], split: str) -> tuple[int, ...]:
    """Select a protocol slice without reordering raw instance IDs."""

    if instance_ids != TURNING_ON_RADIO_PUBLIC_IDS:
        raise RuntimeError("instance order is not the pinned CSV order")
    if split == "official_first10":
        return instance_ids[:10]
    if split == "holdback_last10":
        return instance_ids[10:]
    if split == "all_public":
        return instance_ids
    raise ValueError(f"unknown split: {split}")


def _git(repo_root: Path, *arguments: str, check: bool = True) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        if check:
            raise
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    requested = path.expanduser().absolute()
    resolved = requested.resolve()
    stat = resolved.stat()
    return {
        "path": str(requested),
        "resolved_path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(resolved),
    }


def _checkout_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    top = _git(resolved, "rev-parse", "--show-toplevel", check=False)
    commit = _git(resolved, "rev-parse", "HEAD", check=False)
    status = _git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        check=False,
    )
    if top is None or commit is None or status is None:
        raise RuntimeError(f"required checkout is not readable as git: {resolved}")
    top_path = Path(top).resolve()
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=resolved,
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=resolved,
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout.split(b"\0")
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot fingerprint checkout content: {resolved}"
        ) from error
    content_digest = hashlib.sha256(diff)
    for raw_relative in sorted(item for item in untracked if item):
        relative = os.fsdecode(raw_relative)
        candidate = top_path / relative
        content_digest.update(raw_relative)
        content_digest.update(b"\0")
        if candidate.is_symlink():
            content_digest.update(b"symlink\0")
            content_digest.update(os.fsencode(os.readlink(candidate)))
        elif candidate.is_file():
            content_digest.update(b"file\0")
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content_digest.update(chunk)
        else:
            content_digest.update(b"other\0")
    return {
        "path": str(resolved),
        "toplevel": str(top_path),
        "commit": commit,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_content_sha256": content_digest.hexdigest(),
    }


def source_identity(repo_root: Path) -> dict[str, Any]:
    """Require a clean, committed checkout and return its immutable identity."""

    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("formal serial evaluation requires a clean worktree")
    return {
        "commit": commit,
        "branch": branch,
        "worktree": str(repo_root.resolve()),
        "worktree_dirty": False,
    }


def _validate_entry_python(python: Path, *, repo_root: Path) -> None:
    """Fail before plan creation unless the frozen RPent SDK Python is usable."""

    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import httpx, openai_codex; "
                "import rpent.cli.main; "
                "import robots.behavior.runtime_provider"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = detail[-1] if detail else f"exit {completed.returncode}"
        raise RuntimeError(
            f"RPent entry Python dependency preflight failed: {python}: {suffix}"
        )


def _normalize_cuda_device(cuda_device: str) -> str:
    raw = str(cuda_device).strip()
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError("cuda_device must be one decimal GPU ordinal")
    return str(int(raw))


def _gpu_lock_path(cuda_device: str) -> Path:
    normalized = _normalize_cuda_device(cuda_device)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"rpent-behavior-eval-gpu-{digest}.lock"


def _verify_input_fingerprints(
    *,
    repo_root: Path,
    source: dict[str, Any],
    global_inputs: dict[str, dict[str, Any]],
    entry: EvalEntry,
) -> list[str]:
    errors: list[str] = []
    try:
        current_source = source_identity(repo_root)
    except Exception as error:
        errors.append(f"source identity unavailable: {type(error).__name__}: {error}")
    else:
        if current_source != source:
            errors.append("RPent source identity changed after plan creation")
    for label, expected in global_inputs.items():
        try:
            if label.endswith("_checkout"):
                actual = _checkout_identity(Path(expected["path"]))
            else:
                actual = _file_fingerprint(Path(expected["path"]))
        except Exception as error:
            errors.append(
                f"{label} fingerprint unavailable: {type(error).__name__}: {error}"
            )
            continue
        if actual != expected:
            errors.append(f"{label} changed after plan creation")
    try:
        actual_state = _file_fingerprint(entry.instance_state_path)
    except Exception as error:
        errors.append(
            f"instance state fingerprint unavailable: {type(error).__name__}: {error}"
        )
    else:
        if actual_state["sha256"] != entry.instance_state_sha256:
            errors.append("instance state changed after plan creation")
    return errors


def build_entry_argv(
    *,
    python: Path,
    repo_root: Path,
    output_dir: Path,
    behavior_repo: Path,
    behavior_python: Path,
    checkpoint: Path,
    activity_instance_id: int,
    seed: int,
    cuda_device: str,
    model: str | None,
    max_turns: int,
    cerebrum_timeout_s: int,
) -> tuple[str, ...]:
    """Build one fixed fresh-process Codex SDK invocation."""

    argv = [
        str(python),
        "-m",
        "rpent.cli.main",
        "--env",
        "behavior",
        "--cerebrum",
        "codex",
        "--behavior-control-mode",
        "pi0_nav_pick_vla",
        "--behavior-stage3-press",
        "--suite",
        "behavior_2025_challenge",
        "--task",
        str(TURNING_ON_RADIO_TASK_ID),
        "--task-name",
        TURNING_ON_RADIO_TASK_NAME,
        "--activity-definition-id",
        "0",
        "--activity-instance-id",
        str(activity_instance_id),
        "--scene-model",
        "house_double_floor_lower",
        "--seed",
        str(seed),
        "--max-episode-steps",
        "24756",
        "--behavior-repo",
        str(behavior_repo),
        "--behavior-python",
        str(behavior_python),
        "--policy-checkpoint",
        str(checkpoint),
        "--behavior-pi0-pick-instruction",
        TASK_INSTRUCTION,
        "--cuda-device",
        str(cuda_device),
        "--max-turns",
        str(max_turns),
        "--cerebrum-timeout-s",
        str(cerebrum_timeout_s),
        "--output-dir",
        str(output_dir),
    ]
    if model:
        argv.extend(["--model", model])
    if "--no-driver" in argv or "--vla-endpoint" in argv or "--env-port" in argv:
        raise AssertionError("formal serial entries must own fresh env and VLA servers")
    if repo_root.resolve() != Path(repo_root).resolve():
        raise AssertionError("repo_root must be resolved")
    return tuple(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _proc_stat(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any] | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": int(pid),
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def _owned_group_members(
    process: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Find live members of one manifest-bound dedicated process session."""

    if process.get("managed") is not True:
        return ()
    pid = process.get("pid")
    pgid = process.get("pgid")
    sid = process.get("sid")
    start_ticks = process.get("start_ticks")
    if not all(isinstance(value, int) and value > 0 for value in (pid, pgid, sid)):
        return ()
    if not isinstance(start_ticks, int) or start_ticks <= 0:
        return ()
    if pid != pgid or pid != sid:
        return ()
    leader = _proc_stat(pid, proc_root=proc_root)
    if (
        leader is None
        or leader["state"] == "Z"
        or leader["pgid"] != pgid
        or leader["sid"] != sid
        or leader["start_ticks"] != start_ticks
    ):
        return ()
    return _matching_group_members(
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        proc_root=proc_root,
    )


def _matching_group_members(
    *, pgid: int, sid: int, start_ticks: int, proc_root: Path
) -> tuple[int, ...]:
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return ()
    members: list[int] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _proc_stat(int(entry.name), proc_root=proc_root)
        if (
            stat is not None
            and stat["state"] != "Z"
            and stat["pgid"] == pgid
            and stat["sid"] == sid
            and stat["start_ticks"] >= start_ticks
        ):
            members.append(stat["pid"])
    return tuple(sorted(members))


def _unverified_group_members(
    process: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Report possible recycled/leaderless members but never authorize signals."""

    pid = process.get("pid")
    pgid = process.get("pgid")
    sid = process.get("sid")
    start_ticks = process.get("start_ticks")
    if process.get("managed") is not True or not all(
        isinstance(value, int) and value > 0 for value in (pid, pgid, sid, start_ticks)
    ):
        return ()
    if pid != pgid or pid != sid or _owned_group_members(process, proc_root=proc_root):
        return ()
    return _matching_group_members(
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        proc_root=proc_root,
    )


def _manifest_owned_groups(output_dir: Path) -> dict[str, tuple[int, ...]]:
    manifest = _read_json(output_dir / "run_manifest.json") or {}
    alive: dict[str, tuple[int, ...]] = {}
    for role, process in (manifest.get("processes") or {}).items():
        if not isinstance(process, dict):
            continue
        members = _owned_group_members(process)
        if members:
            alive[str(role)] = members
    return alive


def _manifest_unverified_groups(output_dir: Path) -> dict[str, tuple[int, ...]]:
    manifest = _read_json(output_dir / "run_manifest.json") or {}
    alive: dict[str, tuple[int, ...]] = {}
    for role, process in (manifest.get("processes") or {}).items():
        if not isinstance(process, dict):
            continue
        members = _unverified_group_members(process)
        if members:
            alive[str(role)] = members
    return alive


def _terminate_manifest_processes(
    output_dir: Path, *, timeout_s: float = 30.0
) -> dict[str, tuple[int, ...]]:
    """Stop only manifest-bound dedicated groups and report any survivors."""

    manifest = _read_json(output_dir / "run_manifest.json") or {}
    records = {
        str(role): process
        for role, process in (manifest.get("processes") or {}).items()
        if isinstance(process, dict)
    }
    groups: dict[int, dict[str, Any]] = {}
    for process in records.values():
        members = _owned_group_members(process)
        pgid = process.get("pgid")
        if members and isinstance(pgid, int) and pgid != os.getpgrp():
            groups[pgid] = process
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0.0, timeout_s)
    while groups and time.monotonic() < deadline:
        groups = {
            pgid: process
            for pgid, process in groups.items()
            if _owned_group_members(process)
        }
        if groups:
            time.sleep(0.1)
    for pgid, process in tuple(groups.items()):
        if not _owned_group_members(process):
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + min(5.0, max(0.0, timeout_s))
    while groups and time.monotonic() < kill_deadline:
        groups = {
            pgid: process
            for pgid, process in groups.items()
            if _owned_group_members(process)
        }
        if groups:
            time.sleep(0.1)
    return _manifest_owned_groups(output_dir)


def _terminal_press_wrist_image(
    output_dir: Path, *, expected_entry: EvalEntry | None = None
) -> str | None:
    trace_path = output_dir / "pi0_nav_pick_tool_trace.jsonl"
    try:
        trace = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        trace = [record for record in trace if isinstance(record, dict)]
    except (OSError, json.JSONDecodeError):
        return None
    root = output_dir.resolve()

    def contained_file(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        if candidate.is_symlink():
            return None
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_file() else None

    def valid_png(path: Path, expected: dict[str, Any] | None = None) -> bool:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            return False
        if width < 2 or height < 2:
            return False
        if isinstance(expected, dict):
            if expected.get("sha256") != _sha256(path):
                return False
            if expected.get("width") != width or expected.get("height") != height:
                return False
        return True

    def external_stage3_press_hand(
        *, trace_records: list[dict[str, Any]], hold_step: int
    ) -> str | None:
        checkpoint1 = contained_file(
            str(output_dir / "state_checkpoints" / "state_checkpoint_1.json")
        )
        checkpoint2 = contained_file(
            str(output_dir / "state_checkpoints" / "state_checkpoint_2.json")
        )
        if checkpoint1 is None or checkpoint2 is None:
            return None
        try:
            checkpoint1_payload = json.loads(checkpoint1.read_text(encoding="utf-8"))
            payload = json.loads(checkpoint2.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(checkpoint1_payload, dict) or not isinstance(payload, dict):
            return None
        held_hand = payload.get("held_hand")
        press_hand = payload.get("press_hand")
        prepress = payload.get("prepress")
        projection = (
            prepress.get("button_projection") if isinstance(prepress, dict) else None
        )
        projection_id = (
            projection.get("projection_id") if isinstance(projection, dict) else None
        )
        gate = prepress.get("button_gate") if isinstance(prepress, dict) else None
        projection_metrics = (
            projection.get("projection_metrics")
            if isinstance(projection, dict)
            else None
        )
        checkpoint_env_step = payload.get("env_step")
        expected_wrist = (
            f"{press_hand}_wrist" if press_hand in {"left", "right"} else None
        )
        if (
            checkpoint1_payload.get("schema_version") != 1
            or checkpoint1_payload.get("kind") != "robot_motion_checkpoint"
            or checkpoint1_payload.get("not_simulator_restore") is not True
            or checkpoint1_payload.get("checkpoint_name") != "state_checkpoint_1"
            or checkpoint1_payload.get("stage") != "post_pi0_nav_pick"
            or payload.get("schema_version") != 1
            or payload.get("kind") != "robot_motion_checkpoint"
            or payload.get("not_simulator_restore") is not True
            or payload.get("checkpoint_name") != "state_checkpoint_2"
            or payload.get("stage") != "pre_press_alignment"
            or held_hand not in {"left", "right"}
            or press_hand not in {"left", "right"}
            or held_hand == press_hand
            or checkpoint1_payload.get("held_hand") != held_hand
            or checkpoint1_payload.get("press_hand") != press_hand
            or checkpoint1_payload.get("object_name") != payload.get("object_name")
            or not isinstance(payload.get("object_name"), str)
            or not payload["object_name"]
            or checkpoint1_payload.get("run_binding") != payload.get("run_binding")
            or not isinstance(projection_id, str)
            or not projection_id
            or not isinstance(prepress, dict)
            or prepress.get("source_checkpoint_sha256") != _sha256(checkpoint1)
            or contained_file(prepress.get("source_checkpoint_path")) != checkpoint1
            or not isinstance(checkpoint_env_step, int)
            or isinstance(checkpoint_env_step, bool)
            or not isinstance(gate, dict)
            or gate.get("button_visible") is not True
            or gate.get("face_class") != "BUTTON_FACE"
            or gate.get("positive_signature_complete") is not True
            or projection.get("camera") != "press_wrist"
            or projection.get("resolved_camera") != expected_wrist
            or not isinstance(projection.get("frame_id"), str)
            or not projection["frame_id"]
            or not isinstance(projection.get("capture_group_id"), str)
            or not projection["capture_group_id"]
            or projection.get("env_step") != checkpoint_env_step
            or not isinstance(projection.get("gate_id"), str)
            or not projection["gate_id"]
            or gate.get("camera") != projection.get("camera")
            or gate.get("resolved_camera") != projection.get("resolved_camera")
            or gate.get("frame_id") != projection.get("frame_id")
            or gate.get("capture_group_id") != projection.get("capture_group_id")
            or gate.get("env_step") != projection.get("env_step")
            or gate.get("gate_id") != projection.get("gate_id")
            or not isinstance(projection_metrics, dict)
            or projection_metrics.get("camera") != expected_wrist
            or projection_metrics.get("frame_id") != projection.get("frame_id")
            or projection_metrics.get("step_index") != checkpoint_env_step
        ):
            return None
        if expected_entry is not None:
            binding = payload.get("run_binding")
            expected_binding = {
                "suite": "behavior_2025_challenge",
                "task": TURNING_ON_RADIO_TASK_ID,
                "task_name": TURNING_ON_RADIO_TASK_NAME,
                "activity_definition_id": 0,
                "activity_instance_id": expected_entry.activity_instance_id,
                "scene_model": "house_double_floor_lower",
                "seed": expected_entry.seed,
            }
            if not isinstance(binding, dict) or any(
                binding.get(field) != value for field, value in expected_binding.items()
            ):
                return None
            if not isinstance(binding.get("nonce"), str) or not binding["nonce"]:
                return None
        checkpoint2_sha = _sha256(checkpoint2)
        save_steps = []
        for record in trace_records:
            result = record.get("result")
            step = record.get("step")
            if (
                record.get("tool") == "save_robot_state_checkpoint"
                and isinstance(result, dict)
                and isinstance(step, int)
                and not isinstance(step, bool)
                and step < hold_step
                and result.get("state_checkpoint_2_sha256") == checkpoint2_sha
                and result.get("held_hand") == held_hand
                and result.get("press_hand") == press_hand
            ):
                saved_path = contained_file(result.get("state_checkpoint_2_path"))
                if saved_path == checkpoint2:
                    save_steps.append(step)
        if not save_steps:
            return None
        save_step = max(save_steps)
        for record in trace_records:
            result = record.get("result")
            step = record.get("step")
            if (
                record.get("tool") == "post_pick_direct_finger_toggle"
                and isinstance(result, dict)
                and result.get("task_success") is True
                and isinstance(step, int)
                and not isinstance(step, bool)
                and save_step < step < hold_step
                and result.get("press_hand") == press_hand
                and result.get("projection_id") == projection_id
            ):
                return str(press_hand)
        return None

    # A raw success reached inside pi0_nav_pick is terminal to the Codex SDK,
    # so the env performs the hold and capture before that one tool returns.
    for record in reversed(trace):
        raw_result = record.get("result")
        result = raw_result if isinstance(raw_result, dict) else {}
        evidence = result.get("terminal_success_evidence")
        if record.get("tool") != "pi0_nav_pick" or not isinstance(evidence, dict):
            continue
        requested = evidence.get("hold_frames_requested")
        executed = evidence.get("hold_frames_executed")
        start = evidence.get("start_env_step")
        end = evidence.get("end_env_step")
        press_hand = evidence.get("press_hand")
        resolved_camera = evidence.get("resolved_camera")
        view = evidence.get("terminal_press_wrist")
        if (
            result.get("task_success") is not True
            or evidence.get("complete") is not True
            or evidence.get("source") != "pi0_nav_pick_internal_terminal_finalize"
            or evidence.get("task_success_before_hold") is not True
            or evidence.get("task_success_after_hold") is not True
            or isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested < 4
            or executed != requested
            or isinstance(start, bool)
            or not isinstance(start, int)
            or end != start + executed
            or press_hand not in {"left", "right"}
            or evidence.get("held_hand") not in {"left", "right"}
            or evidence.get("held_hand") == press_hand
            or evidence.get("role_resolution_source")
            not in {"strict_handoff", "unique_terminal_attachment_evidence"}
            or evidence.get("logical_camera") != "press_wrist"
            or resolved_camera != f"{press_hand}_wrist"
            or not isinstance(view, dict)
            or view.get("camera") != resolved_camera
            or view.get("env_step") != end
            or not isinstance(view.get("frame_id"), str)
            or not view["frame_id"]
            or view.get("capture_group_id") != evidence.get("capture_group_id")
        ):
            continue
        metadata_path = contained_file(evidence.get("metadata_path"))
        image_path = contained_file(view.get("path"))
        if metadata_path is None or image_path is None:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metadata_view = metadata.get("terminal_press_wrist")
        if (
            metadata.get("complete") is not True
            or metadata.get("resolved_camera") != resolved_camera
            or not isinstance(metadata_view, dict)
            or contained_file(metadata_view.get("path")) != image_path
            or metadata_view.get("sha256") != view.get("sha256")
            or metadata_view.get("width") != view.get("width")
            or metadata_view.get("height") != view.get("height")
            or metadata_view.get("frame_id") != view.get("frame_id")
            or metadata_view.get("capture_group_id") != view.get("capture_group_id")
            or metadata_view.get("env_step") != view.get("env_step")
            or not valid_png(image_path, view)
        ):
            continue
        return str(image_path)

    def valid_external_hold(record: dict[str, Any]) -> bool:
        result = record.get("result")
        step = record.get("step")
        hold_input = record.get("input")
        if record.get("tool") != "post_success_hold_frames" or not isinstance(
            result, dict
        ):
            return False
        requested = result.get("requested_frames")
        executed = result.get("executed_frames")
        start = result.get("start_env_step")
        end = result.get("end_env_step")
        return bool(
            result.get("task_success") is True
            and result.get("primitive_success") is True
            and isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(requested, int)
            and not isinstance(requested, bool)
            and requested >= 4
            and isinstance(hold_input, dict)
            and hold_input.get("frames") == requested
            and executed == requested
            and isinstance(start, int)
            and not isinstance(start, bool)
            and end == start + executed
        )

    hold_records = [record for record in trace if valid_external_hold(record)]
    if not hold_records:
        return None
    last_hold = max(hold_records, key=lambda record: int(record.get("step", -1)))
    hold_step = int(last_hold.get("step", -1))
    hold_end_env_step = (last_hold.get("result") or {}).get("end_env_step")
    press_hand = external_stage3_press_hand(trace_records=trace, hold_step=hold_step)
    if press_hand not in {"left", "right"}:
        return None
    for record in reversed(trace):
        raw_result = record.get("result")
        result = raw_result if isinstance(raw_result, dict) else {}
        record_step = record.get("step")
        rgb_path = (result.get("visual_review") or {}).get("rgb_path")
        if (
            record.get("tool") == "observe"
            and (record.get("input") or {}).get("camera") == "press_wrist"
            and result.get("task_success") is True
            and isinstance(record_step, int)
            and not isinstance(record_step, bool)
            and record_step > hold_step
            and result.get("camera") == "press_wrist"
            and result.get("resolved_camera") == f"{press_hand}_wrist"
            and result.get("total_env_steps") == hold_end_env_step
            and isinstance(result.get("frame_id"), str)
            and bool(result.get("frame_id"))
            and isinstance(result.get("capture_group"), dict)
            and isinstance(result["capture_group"].get("id"), str)
            and bool(result["capture_group"]["id"])
        ):
            candidate = contained_file(rgb_path)
            metadata_path = contained_file(
                (result.get("visual_review") or {}).get("metadata_path")
            )
            try:
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path is not None
                    else None
                )
            except (OSError, json.JSONDecodeError):
                metadata = None
            metadata_group = (
                metadata.get("capture_group") if isinstance(metadata, dict) else None
            )
            if (
                candidate is not None
                and valid_png(candidate)
                and isinstance(metadata, dict)
                and contained_file(metadata.get("rgb_path")) == candidate
                and contained_file(metadata.get("metadata_path")) == metadata_path
                and metadata.get("camera") == f"{press_hand}_wrist"
                and metadata.get("frame_id") == result.get("frame_id")
                and isinstance(metadata_group, dict)
                and metadata_group.get("id") == result["capture_group"]["id"]
                and metadata.get("total_env_steps") == hold_end_env_step
            ):
                return str(candidate)
    return None


def _has_valid_post_success_hold(trace: list[dict[str, Any]]) -> bool:
    for record in trace:
        if not isinstance(record, dict):
            continue
        raw_result = record.get("result")
        result = raw_result if isinstance(raw_result, dict) else {}
        if record.get("tool") == "pi0_nav_pick":
            evidence = result.get("terminal_success_evidence")
            if not isinstance(evidence, dict):
                continue
            requested = evidence.get("hold_frames_requested")
            executed = evidence.get("hold_frames_executed")
            start = evidence.get("start_env_step")
            end = evidence.get("end_env_step")
            if (
                result.get("task_success") is True
                and evidence.get("task_success_before_hold") is True
                and evidence.get("task_success_after_hold") is True
                and isinstance(requested, int)
                and not isinstance(requested, bool)
                and requested >= 4
                and executed == requested
                and isinstance(start, int)
                and not isinstance(start, bool)
                and end == start + executed
            ):
                return True
        if record.get("tool") != "post_success_hold_frames":
            continue
        step = record.get("step")
        hold_input = record.get("input")
        requested = result.get("requested_frames")
        executed = result.get("executed_frames")
        start = result.get("start_env_step")
        end = result.get("end_env_step")
        if (
            result.get("task_success") is True
            and result.get("primitive_success") is True
            and isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(requested, int)
            and not isinstance(requested, bool)
            and requested >= 4
            and isinstance(hold_input, dict)
            and hold_input.get("frames") == requested
            and executed == requested
            and isinstance(start, int)
            and not isinstance(start, bool)
            and end == start + executed
        ):
            return True
    return False


def validate_instance_result(
    entry: EvalEntry,
    *,
    source_commit: str,
    subprocess_exit_code: int | None,
    timed_out: bool,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Classify one run from bound raw artifacts, never from exit code alone."""

    errors: list[str] = []
    final_result = _read_json(entry.output_dir / "final_result.json")
    manifest = _read_json(entry.output_dir / "run_manifest.json")
    if timed_out:
        errors.append("top-level RPent process timed out")
    if manifest is None:
        errors.append("missing or invalid run_manifest.json")
    else:
        task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
        expected = {
            "suite": "behavior_2025_challenge",
            "task": TURNING_ON_RADIO_TASK_ID,
            "task_name": TURNING_ON_RADIO_TASK_NAME,
            "activity_definition_id": 0,
            "activity_instance_id": entry.activity_instance_id,
            "activity_instance_dir": str(entry.instance_state_path.parent.resolve()),
            "scene_model": "house_double_floor_lower",
            "seed": entry.seed,
            "max_episode_steps": 24756,
        }
        for field, value in expected.items():
            if task.get(field) != value:
                errors.append(f"manifest task binding mismatch: {field}")
        if manifest.get("control_mode") != "pi0_nav_pick_vla":
            errors.append("manifest control_mode mismatch")
        if manifest.get("stage3_press_enabled") is not True:
            errors.append("manifest stage3 press is not enabled")
        if manifest.get("commit") != source_commit:
            errors.append("manifest source commit mismatch")
        if manifest.get("worktree_dirty") is not False:
            errors.append("manifest source worktree is dirty")
        if manifest.get("status") != "stopped":
            errors.append("manifest lifecycle did not stop cleanly")
        if manifest.get("checkpoint") != str(entry.checkpoint.resolve()):
            errors.append("manifest policy checkpoint mismatch")
        if manifest.get("gpu") != entry.cuda_device:
            errors.append("manifest GPU binding mismatch")
        for role, process in (manifest.get("processes") or {}).items():
            if not isinstance(process, dict):
                errors.append(f"invalid process manifest for {role}")
                continue
            if process.get("managed") is True and process.get("stopped_at") is None:
                errors.append(f"managed {role} process lacks stopped_at")
            members = _owned_group_members(process)
            if members:
                errors.append(
                    f"managed {role} process group is still alive: {list(members)}"
                )
            unverified = _unverified_group_members(process)
            if unverified:
                errors.append(
                    f"managed {role} group identity is ambiguous: {list(unverified)}"
                )
    temporary = sorted(
        str(path)
        for path in (entry.output_dir / "state_checkpoints").glob(
            "tmp_state_checkpoint_*.json*"
        )
    )
    if temporary:
        errors.append("temporary checkpoint JSON was not deleted")
    if final_result is None:
        errors.append("missing or invalid final_result.json")
        if timed_out or subprocess_exit_code not in {0, None}:
            return "run_error", errors, None
        return "incomplete", errors, None
    if subprocess_exit_code not in {0, None}:
        errors.append("top-level RPent process returned nonzero")
    if final_result.get("runtime_cleanup") != "complete":
        errors.append("runtime cleanup did not complete")
    if (
        final_result.get("error") is not None
        or final_result.get("run_status") == "error"
    ):
        errors.append("RPent final_result reports an execution error")
    task_success = final_result.get("task_success")
    if task_success not in {True, False, None}:
        errors.append("task_success is not boolean or null")
    expected_source = (
        'info["done"]["success"]' if task_success in {True, False} else None
    )
    if final_result.get("official_success_source") != expected_source:
        errors.append("invalid official success source")
    if task_success is True:
        trace_path = entry.output_dir / "pi0_nav_pick_tool_trace.jsonl"
        try:
            trace = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            trace = [record for record in trace if isinstance(record, dict)]
        except (OSError, json.JSONDecodeError):
            trace = []
        if not _has_valid_post_success_hold(trace):
            errors.append("successful run lacks post-success render hold")
        terminal_image = _terminal_press_wrist_image(
            entry.output_dir, expected_entry=entry
        )
        if terminal_image is None:
            errors.append("successful run lacks fresh post-hold press-wrist image")
        elif not _green_center_marker_visible(Path(terminal_image)):
            errors.append(
                "successful run terminal press-wrist image does not show green "
                "center marker"
            )
    if errors:
        if (
            timed_out
            or subprocess_exit_code not in {0, None}
            or final_result.get("error")
        ):
            return "run_error", errors, final_result
        return "incomplete", errors, final_result
    if task_success is True:
        return "passed", errors, final_result
    if task_success is False:
        return "task_failed", errors, final_result
    return "incomplete", ["official task_success is missing"], final_result


def _terminate_top_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def _run_entry(
    entry: EvalEntry,
    *,
    repo_root: Path,
    log_stream: BinaryIO,
    timeout_s: int,
) -> tuple[int | None, bool]:
    if entry.output_dir.exists() and any(entry.output_dir.iterdir()):
        raise RuntimeError(f"entry output directory is not empty: {entry.output_dir}")
    if entry.output_dir.exists():
        entry.output_dir.rmdir()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    process = subprocess.Popen(
        entry.argv,
        cwd=repo_root,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        _terminate_manifest_processes(entry.output_dir)
        _terminate_top_process(process)
        _terminate_manifest_processes(entry.output_dir)
        return process.returncode, True
    except BaseException:
        _terminate_manifest_processes(entry.output_dir)
        _terminate_top_process(process)
        _terminate_manifest_processes(entry.output_dir)
        raise


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run turning_on_radio public instances serially with fresh Codex, "
            "env, and VLA processes and no automatic retry."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda-device", required=True)
    parser.add_argument(
        "--split",
        choices=("official_first10", "holdback_last10", "all_public"),
        default="official_first10",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--cerebrum-timeout-s", type=int, default=7200)
    parser.add_argument("--instance-timeout-s", type=int, default=14400)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo", default=None)
    parser.add_argument("--behavior-python", default=None)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--test-instances-csv", default=None)
    parser.add_argument(
        "--expected-csv-sha256",
        default=TEST_INSTANCES_SHA256,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Create an immutable plan, then execute it synchronously exactly once."""

    args = _parse_args(argv)
    if (
        args.max_turns <= 0
        or args.cerebrum_timeout_s <= 0
        or args.instance_timeout_s <= 0
    ):
        raise SystemExit("turn and timeout limits must be positive")
    repo_root = Path(args.repo_root).expanduser().resolve()
    behavior_repo = (
        Path(
            args.behavior_repo
            or os.environ.get("RPENT_RLINF_ROOT")
            or repo_root.parent / "RLinf_agentic_push"
        )
        .expanduser()
        .resolve()
    )
    behavior_python = (
        Path(
            args.behavior_python or behavior_repo / ".venv-behavior" / "bin" / "python"
        )
        .expanduser()
        .absolute()
    )
    entry_python = Path(args.python).expanduser().absolute()
    checkpoint_raw = args.policy_checkpoint or os.environ.get("PI05_CHECKPOINT_PATH")
    if not checkpoint_raw:
        raise SystemExit("--policy-checkpoint or PI05_CHECKPOINT_PATH is required")
    checkpoint = Path(checkpoint_raw).expanduser().resolve()
    metadata_root = (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
    )
    csv_path = (
        Path(
            args.test_instances_csv or metadata_root / "metadata" / "test_instances.csv"
        )
        .expanduser()
        .resolve()
    )
    instance_dir = (
        metadata_root
        / "scenes"
        / "house_double_floor_lower"
        / "json"
        / "house_double_floor_lower_task_turning_on_radio_instances"
    )
    for required in (
        repo_root / "pyproject.toml",
        entry_python,
        behavior_python,
        checkpoint / "model.safetensors",
        checkpoint
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json",
        csv_path,
        instance_dir,
    ):
        if not required.exists():
            raise SystemExit(f"required path is missing: {required}")

    try:
        _validate_entry_python(entry_python, repo_root=repo_root)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    source = source_identity(repo_root)
    public_ids = read_turning_on_radio_instances(
        csv_path, expected_sha256=args.expected_csv_sha256
    )
    selected = select_instances(public_ids, args.split)
    csv_offset = 0 if args.split != "holdback_last10" else 10
    try:
        cuda_device = _normalize_cuda_device(args.cuda_device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    model_path = checkpoint / "model.safetensors"
    norm_stats_path = (
        checkpoint
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json"
    )
    behavior_dataset_repo = behavior_repo / ".venv-behavior" / "BEHAVIOR-1K"
    global_inputs = {
        "model": _file_fingerprint(model_path),
        "norm_stats": _file_fingerprint(norm_stats_path),
        "test_instances_csv": _file_fingerprint(csv_path),
        "rpent_python": _file_fingerprint(entry_python),
        "behavior_python": _file_fingerprint(behavior_python),
        "behavior_checkout": _checkout_identity(behavior_repo),
        "behavior_dataset_checkout": _checkout_identity(behavior_dataset_repo),
    }

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_paths = (
        _gpu_lock_path(cuda_device),
        output_root.parent / f".{output_root.name}.lock",
    )
    lock_streams: list[Any] = []
    try:
        for lock_path in lock_paths:
            stream = lock_path.open("w", encoding="utf-8")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise SystemExit(
                    f"another serial evaluator owns {lock_path}"
                ) from error
            lock_streams.append(stream)
        if output_root.exists() and any(output_root.iterdir()):
            raise SystemExit(f"--output-root must be absent or empty: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)

        entries: list[EvalEntry] = []
        for split_position, instance_id in enumerate(selected):
            csv_position = csv_offset + split_position
            if public_ids[csv_position] != instance_id:
                raise RuntimeError("CSV position and raw instance ID binding diverged")
            state_matches = sorted(
                instance_dir.glob(f"*_0_{instance_id}_template-tro_state.json")
            )
            if len(state_matches) != 1:
                raise RuntimeError(
                    f"expected one tro-state for instance {instance_id}, "
                    f"found {len(state_matches)}"
                )
            instance_state_path = state_matches[0].resolve()
            output_dir = output_root / (
                f"{split_position:02d}_csv{csv_position:02d}_"
                f"instance{instance_id}_seed{args.seed}"
            )
            entry_argv = build_entry_argv(
                python=entry_python,
                repo_root=repo_root,
                output_dir=output_dir,
                behavior_repo=behavior_repo,
                behavior_python=behavior_python,
                checkpoint=checkpoint,
                activity_instance_id=instance_id,
                seed=args.seed,
                cuda_device=cuda_device,
                model=args.model,
                max_turns=args.max_turns,
                cerebrum_timeout_s=args.cerebrum_timeout_s,
            )
            entries.append(
                EvalEntry(
                    split_position=split_position,
                    csv_position=csv_position,
                    activity_instance_id=instance_id,
                    seed=args.seed,
                    output_dir=output_dir,
                    argv=entry_argv,
                    checkpoint=checkpoint,
                    cuda_device=cuda_device,
                    instance_state_path=instance_state_path,
                    instance_state_sha256=_sha256(instance_state_path),
                )
            )

        plan_path = output_root / "eval_plan.json"
        results_path = output_root / "eval_results.jsonl"
        plan = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "protocol": {
                "task_id": TURNING_ON_RADIO_TASK_ID,
                "task_name": TURNING_ON_RADIO_TASK_NAME,
                "split": args.split,
                "seed": args.seed,
                "model": args.model,
                "cuda_device": cuda_device,
                "gpu_lock": str(lock_paths[0]),
                "max_parallel": 1,
                "max_attempts_per_instance": 1,
                "automatic_retry": False,
                "fresh_top_level_process_per_instance": True,
                "fresh_codex_context_per_instance": True,
                "cross_instance_adaptation": False,
                "success_field": (
                    'final_result.task_success from info["done"]["success"]'
                ),
                "restore_policy": ("same-run robot-state JSON via guarded CuRobo only"),
                "persistent_checkpoints": [
                    "state_checkpoint_1.json",
                    "state_checkpoint_2.json",
                ],
                "max_temporary_checkpoints": 4,
                "temporary_checkpoint_cleanup_required": True,
                "terminal_green_marker_required": True,
                "external_stage3_checkpoint2_lineage_required": True,
            },
            "source": source,
            "input_fingerprints": global_inputs,
            "entries": [
                {
                    "split_position": entry.split_position,
                    "csv_position": entry.csv_position,
                    "activity_instance_id": entry.activity_instance_id,
                    "seed": entry.seed,
                    "output_dir": str(entry.output_dir),
                    "argv": list(entry.argv),
                    "instance_state": {
                        "path": str(entry.instance_state_path),
                        "sha256": entry.instance_state_sha256,
                    },
                }
                for entry in entries
            ],
        }
        _atomic_json(plan_path, plan)

        launcher_logs = output_root / "launcher_logs"
        launcher_logs.mkdir()
        abort_remaining = False
        interrupted = False
        for entry in entries:
            if abort_remaining:
                _append_jsonl(
                    results_path,
                    {
                        "activity_instance_id": entry.activity_instance_id,
                        "csv_position": entry.csv_position,
                        "outcome": "not_run",
                        "reason": (
                            "serial ownership or immutable inputs could not be "
                            "guaranteed after prior run"
                        ),
                    },
                )
                continue
            started_at = _utc_now()
            started = time.monotonic()
            log_path = (
                launcher_logs
                / f"csv{entry.csv_position:02d}_i{entry.activity_instance_id}.log"
            )
            exit_code: int | None = None
            timed_out = False
            launch_error: str | None = None
            try:
                fingerprint_errors = _verify_input_fingerprints(
                    repo_root=repo_root,
                    source=source,
                    global_inputs=global_inputs,
                    entry=entry,
                )
            except BaseException as error:
                fingerprint_errors = []
                launch_error = f"{type(error).__name__}: {error}"
                interrupted = not isinstance(error, Exception)
                abort_remaining = True
            if launch_error is not None:
                pass
            elif fingerprint_errors:
                launch_error = "; ".join(fingerprint_errors)
                abort_remaining = True
            else:
                try:
                    with log_path.open("wb") as log_stream:
                        exit_code, timed_out = _run_entry(
                            entry,
                            repo_root=repo_root,
                            log_stream=log_stream,
                            timeout_s=args.instance_timeout_s,
                        )
                except BaseException as error:
                    launch_error = f"{type(error).__name__}: {error}"
                    interrupted = not isinstance(error, Exception)
                    abort_remaining = interrupted

            alive_before_cleanup = _manifest_owned_groups(entry.output_dir)
            alive_after_cleanup: dict[str, tuple[int, ...]] = {}
            if alive_before_cleanup:
                alive_after_cleanup = _terminate_manifest_processes(entry.output_dir)
                if launch_error is None:
                    launch_error = (
                        "managed process groups required forced cleanup: "
                        + ", ".join(sorted(alive_before_cleanup))
                    )
            if alive_after_cleanup:
                abort_remaining = True
            ambiguous_groups = _manifest_unverified_groups(entry.output_dir)
            if ambiguous_groups:
                abort_remaining = True
                if launch_error is None:
                    launch_error = (
                        "managed process identity became ambiguous; refusing to "
                        "signal or continue serial evaluation: "
                        + ", ".join(sorted(ambiguous_groups))
                    )
            outcome, validation_errors, final_result = validate_instance_result(
                entry,
                source_commit=str(source["commit"]),
                subprocess_exit_code=exit_code,
                timed_out=timed_out,
            )
            if launch_error is not None:
                outcome = "run_error"
                validation_errors.insert(0, launch_error)
            record = {
                "schema_version": 1,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_s": round(time.monotonic() - started, 3),
                "split_position": entry.split_position,
                "csv_position": entry.csv_position,
                "activity_instance_id": entry.activity_instance_id,
                "seed": entry.seed,
                "attempt": 1,
                "subprocess_exit_code": exit_code,
                "timed_out": timed_out,
                "outcome": outcome,
                "task_success": (
                    final_result.get("task_success")
                    if isinstance(final_result, dict)
                    else None
                ),
                "validation_errors": validation_errors,
                "forced_cleanup_groups": {
                    role: list(members)
                    for role, members in alive_before_cleanup.items()
                },
                "alive_managed_groups": {
                    role: list(members) for role, members in alive_after_cleanup.items()
                },
                "ambiguous_managed_groups": {
                    role: list(members) for role, members in ambiguous_groups.items()
                },
                "terminal_press_wrist_image": _terminal_press_wrist_image(
                    entry.output_dir
                ),
                "output_dir": str(entry.output_dir),
                "launcher_log": str(log_path),
            }
            _append_jsonl(results_path, record)

        results = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = {
            "schema_version": 1,
            "finished_at": _utc_now(),
            "plan_path": str(plan_path),
            "results_path": str(results_path),
            "interrupted": interrupted,
            "counts": {
                outcome: sum(item.get("outcome") == outcome for item in results)
                for outcome in (
                    "passed",
                    "task_failed",
                    "run_error",
                    "incomplete",
                    "not_run",
                )
            },
        }
        _atomic_json(output_root / "eval_summary.json", summary)
        if interrupted:
            return 130
        return 0 if all(item.get("outcome") == "passed" for item in results) else 1
    finally:
        for stream in reversed(lock_streams):
            try:
                fcntl.flock(stream, fcntl.LOCK_UN)
            finally:
                stream.close()


__all__ = [
    "EvalEntry",
    "TEST_INSTANCES_SHA256",
    "TURNING_ON_RADIO_PUBLIC_IDS",
    "build_entry_argv",
    "main",
    "read_turning_on_radio_instances",
    "select_instances",
    "validate_instance_result",
]
