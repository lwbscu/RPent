"""Finite, artifact-local BEHAVIOR candidate Explore jobs.

This temporary campaign wrapper never changes the canonical public-seed
mapping or the canonical publication format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
)
from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH
from robots.behavior.serial_explore import (
    ATTEMPT_TOOL_CALLS,
    ExploreConfig,
    _expected_job_checkpoint_binding,
    _official_success_binding,
    run_explore_job,
)
from robots.behavior.task_specs import (
    TURNING_ON_RADIO_TASK_SPEC,
    get_task_spec,
)
from robots.behavior.toolkit import BehaviorToolkit

CANDIDATE_MAX_WALL_CLOCK_S = 6900
CANDIDATE_PLANNER_TIMEOUT_S = 6900
CANDIDATE_ATTEMPT_TIMEOUT_S = 7200
_EVIDENCE_NAMES = (
    "behavior_action_trace.jsonl",
    "behavior_tool_trace.jsonl",
    "final_result.json",
    "run_manifest.json",
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_state_binding(
    config: ExploreConfig,
    candidate_state_file: Path,
) -> Path:
    task_spec = get_task_spec(config.task_name)
    state_file = Path(candidate_state_file).expanduser().resolve(strict=True)
    if not state_file.is_file() or state_file.is_symlink():
        raise ValueError("candidate state must be a non-symlink regular file")
    if state_file.parent.name != task_spec.state_dir_name:
        raise ValueError("candidate state directory does not match the selected task")
    expected_suffix = f"_{int(config.candidate_instance_id)}_template-tro_state.json"
    if not state_file.name.endswith(expected_suffix):
        raise ValueError("candidate state filename does not match the instance")
    if _sha256(state_file) != config.candidate_state_sha256:
        raise ValueError("candidate state SHA256 changed before launch")
    return state_file


def build_candidate_config(
    *,
    output_root: Path,
    repo_root: Path,
    python: Path,
    behavior_repo: Path,
    behavior_python: Path,
    policy_checkpoint: Path,
    candidate_instance_id: int,
    candidate_state_file: Path,
    resource_binding: DatasetResourceBinding,
    task_name: str = TURNING_ON_RADIO_TASK_SPEC.task_name,
    public_seed: int = 0,
    policy_checkpoint_binding: dict[str, Any] | None = None,
    cuda_device: str = "7",
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    max_turns: int = 1000,
    max_tool_calls: int = ATTEMPT_TOOL_CALLS,
    planner_timeout_s: int = CANDIDATE_PLANNER_TIMEOUT_S,
    attempt_timeout_s: int = CANDIDATE_ATTEMPT_TIMEOUT_S,
    max_wall_clock_s: int = CANDIDATE_MAX_WALL_CLOCK_S,
    vla_ready_timeout_s: int = 1800,
    min_free_disk_gb: float = 10.0,
    dashboard: bool = False,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8765,
) -> ExploreConfig:
    """Build one finite candidate-only Explore configuration."""

    task_spec = get_task_spec(task_name)
    if (
        int(candidate_instance_id) <= 0
        or task_spec.classify_instance(int(candidate_instance_id)).kind != "candidate"
    ):
        raise ValueError(
            "candidate instance must be positive and non-public for the selected task"
        )
    if str(cuda_device) != "7":
        raise ValueError("candidate BEHAVIOR Explore is restricted to GPU7")
    if model != "gpt-5.5" or reasoning_effort != "xhigh":
        raise ValueError("this campaign requires gpt-5.5 with xhigh reasoning")
    if int(max_turns) <= 0 or int(max_tool_calls) <= 0:
        raise ValueError("candidate turn and tool-call budgets must be positive")
    for label, value in (
        ("planner timeout", planner_timeout_s),
        ("attempt timeout", attempt_timeout_s),
        ("wall-clock budget", max_wall_clock_s),
    ):
        if int(value) <= 0 or int(value) > 7200:
            raise ValueError(f"candidate {label} must be in 1..7200 seconds")
    state_file = Path(candidate_state_file).expanduser().resolve(strict=True)
    if not state_file.is_file() or state_file.is_symlink():
        raise ValueError("candidate state must be a non-symlink regular file")
    if state_file.parent.name != task_spec.state_dir_name:
        raise ValueError("candidate state directory does not match the selected task")
    expected_suffix = f"_{int(candidate_instance_id)}_template-tro_state.json"
    if not state_file.name.endswith(expected_suffix):
        raise ValueError("candidate state filename does not match the instance")
    checkpoint = Path(policy_checkpoint).expanduser().resolve(strict=True)
    expected_checkpoint_binding = _expected_job_checkpoint_binding(checkpoint)
    if (
        policy_checkpoint_binding is not None
        and policy_checkpoint_binding != expected_checkpoint_binding
    ):
        raise ValueError("candidate checkpoint binding differs from the shared profile")
    if not isinstance(resource_binding, DatasetResourceBinding):
        raise ValueError("candidate Explore requires pinned BEHAVIOR resources")
    if resource_binding.subtree != "behavior":
        raise ValueError("candidate resource binding must select behavior")
    verify_pinned_dataset_resources(resource_binding)
    return ExploreConfig(
        output_root=Path(output_root).expanduser().resolve(),
        repo_root=Path(repo_root).expanduser().resolve(),
        python=Path(python).expanduser().absolute(),
        behavior_repo=Path(behavior_repo).expanduser().resolve(),
        behavior_python=Path(behavior_python).expanduser().absolute(),
        policy_checkpoint=checkpoint,
        policy_checkpoint_binding=expected_checkpoint_binding,
        task_name=task_spec.task_name,
        public_seed=int(public_seed),
        cuda_device="7",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_turns=int(max_turns),
        max_tool_calls=int(max_tool_calls),
        planner_timeout_s=int(planner_timeout_s),
        attempt_timeout_s=int(attempt_timeout_s),
        max_wall_clock_s=int(max_wall_clock_s),
        vla_ready_timeout_s=int(vla_ready_timeout_s),
        min_free_disk_gb=float(min_free_disk_gb),
        dashboard=bool(dashboard),
        dashboard_host=str(dashboard_host),
        dashboard_port=int(dashboard_port),
        dashboard_language="en",
        resume=False,
        candidate_instance_id=int(candidate_instance_id),
        candidate_state_sha256=_sha256(state_file),
        max_attempts=1,
        resource_binding=resource_binding,
    )


def _attempt_dir(root: Path, manifest: dict[str, Any]) -> Path | None:
    attempts = manifest.get("attempts")
    candidate: Path | None = None
    if isinstance(attempts, list) and len(attempts) == 1:
        attempt = attempts[0]
        value = attempt.get("output_dir") if isinstance(attempt, dict) else None
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate
    if candidate is None:
        protocol = manifest.get("protocol")
        task_name = (
            protocol.get("task_name")
            if isinstance(protocol, dict) and isinstance(protocol.get("task_name"), str)
            else TURNING_ON_RADIO_TASK_SPEC.task_name
        )
        discovered = list(
            (
                root
                / "attempts"
                / (
                    f"{task_name}_candidate_i"
                    f"{manifest.get('native_binding', {}).get('activity_instance_id', '')}"
                )
            ).glob("attempt_*")
        )
        if len(discovered) != 1:
            return None
        candidate = discovered[0]
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_dir() and not resolved.is_symlink() else None


def _copy_or_empty(source: Path | None, target: Path) -> None:
    if source is not None and source.is_file() and not source.is_symlink():
        _atomic_bytes(target, source.read_bytes())
    elif not target.exists():
        _atomic_bytes(target, b"")


def _fallback_payload(
    config: ExploreConfig,
    *,
    error: str,
    runtime_started: bool,
    task_success: bool | None,
) -> dict[str, Any]:
    task_spec = get_task_spec(config.task_name)
    return {
        "schema_version": 1,
        "campaign_kind": "candidate_instance_explore",
        "outer_harness_fallback": True,
        "runtime_started": runtime_started,
        "task_success": task_success,
        "official_success_source": None,
        "publication_complete": False,
        "task_identity": {
            "task_name": task_spec.task_name,
            "activity_definition_id": task_spec.activity_definition_id,
            "activity_instance_id": config.candidate_instance_id,
        },
        "activity_instance_id": config.candidate_instance_id,
        "state_sha256": config.candidate_state_sha256,
        "resource_source": (
            config.resource_binding.as_dict()
            if isinstance(config.resource_binding, DatasetResourceBinding)
            else None
        ),
        "error": error,
        "sealed_at": _utc_now(),
    }


def _manifest_matches_candidate_identity(
    config: ExploreConfig,
    manifest: dict[str, Any],
) -> bool:
    task_spec = get_task_spec(config.task_name)
    protocol = manifest.get("protocol")
    native = manifest.get("native_binding")
    identity = manifest.get("task_identity")
    expected_identity = {
        "task_name": task_spec.task_name,
        "activity_definition_id": task_spec.activity_definition_id,
        "activity_instance_id": config.candidate_instance_id,
    }
    return bool(
        isinstance(protocol, dict)
        and protocol.get("task_index") == task_spec.task_index
        and protocol.get("task_name") == task_spec.task_name
        and protocol.get("public_seed") == config.public_seed
        and isinstance(native, dict)
        and native.get("mapping_version") == task_spec.candidate_mapping_version
        and native.get("activity_definition_id") == task_spec.activity_definition_id
        and native.get("activity_instance_id") == config.candidate_instance_id
        and native.get("state_sha256") == config.candidate_state_sha256
        and identity == expected_identity
    )


def _seal_root_evidence(
    config: ExploreConfig,
    manifest: dict[str, Any],
    *,
    error: str | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    session_path = root / "session_manifest.json"
    if not session_path.is_file():
        _atomic_json(session_path, manifest)
    attempt_dir = _attempt_dir(root, manifest)
    binding = (
        _official_success_binding(attempt_dir) if attempt_dir is not None else None
    )
    claimed_success = manifest.get("task_success") is True
    runtime_started = attempt_dir is not None
    if binding is not None and not _manifest_matches_candidate_identity(
        config,
        manifest,
    ):
        binding = None
        error = "candidate success evidence has a mismatched task identity"
    if binding is None and claimed_success and error is None:
        error = "candidate claimed success without a valid raw-success receipt"
    if binding is not None:
        fallback_success: bool | None = True
    elif runtime_started:
        child_final = _read_json(attempt_dir / "final_result.json")
        fallback_success = (
            False
            if isinstance(child_final, dict)
            and child_final.get("task_success") is False
            else None
        )
    else:
        fallback_success = None
    fallback = _fallback_payload(
        config,
        error=error or "candidate runtime did not produce complete evidence",
        runtime_started=runtime_started,
        task_success=fallback_success,
    )
    for name in ("behavior_action_trace.jsonl", "behavior_tool_trace.jsonl"):
        _copy_or_empty(
            attempt_dir / name if attempt_dir is not None else None,
            root / name,
        )
    for name in ("final_result.json", "run_manifest.json"):
        target = root / name
        source = attempt_dir / name if attempt_dir is not None else None
        if (
            binding is None
            and claimed_success
            and source is not None
            and source.is_file()
            and not source.is_symlink()
        ):
            _copy_or_empty(source, root / f"unverified_child_{name}")
            _atomic_json(target, fallback)
        elif source is not None and source.is_file() and not source.is_symlink():
            _copy_or_empty(source, target)
        elif not target.exists():
            _atomic_json(target, fallback)
    terminal_failure_path = (
        attempt_dir / "terminal_failure_receipt.json"
        if attempt_dir is not None
        else None
    )
    if (
        terminal_failure_path is not None
        and terminal_failure_path.is_file()
        and not terminal_failure_path.is_symlink()
    ):
        _copy_or_empty(
            terminal_failure_path,
            root / "terminal_failure_receipt.json",
        )
    return attempt_dir, binding


def _publish_candidate(
    config: ExploreConfig,
    *,
    state_file: Path,
    attempt_dir: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    task_spec = get_task_spec(config.task_name)
    root = config.output_root
    source_paths = {
        "official_success_receipt": attempt_dir / "official_success_receipt.json",
        "behavior_action_trace": root / "behavior_action_trace.jsonl",
        "behavior_tool_trace": root / "behavior_tool_trace.jsonl",
        "final_result": root / "final_result.json",
        "run_manifest": root / "run_manifest.json",
        "session_manifest": root / "session_manifest.json",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    publication = root / "candidate_publication"
    if publication.exists():
        raise RuntimeError("candidate publication already exists")
    stage = root / f".candidate_publication.{secrets.token_hex(8)}.tmp"
    stage.mkdir()
    try:
        tool_trace = [
            json.loads(line)
            for line in (root / "behavior_tool_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if not tool_trace or not all(isinstance(record, dict) for record in tool_trace):
            raise RuntimeError("candidate publication requires a valid tool trace")
        derivation = stage / "_derivation"
        derivation.mkdir()
        session = _read_json(root / "session_manifest.json") or {}
        fake = object.__new__(BehaviorToolkit)
        fake._tool_trace = tool_trace
        fake._primitives = SimpleNamespace(
            attempt_index=int(binding["attempt_index"]),
            output_dir=derivation,
            job_id=session.get("job_id"),
            attempt_nonce=binding["attempt_nonce"],
            run_nonce=binding["run_nonce"],
            task_name=task_spec.task_name,
            task_spec=task_spec,
            public_seed=config.public_seed,
        )
        records = fake._symbolic_recipe()
        BehaviorToolkit.validate_symbolic_publication(records)
        recipe_path = derivation / "recipe.jsonl"
        BehaviorToolkit._write_json_atomic(
            recipe_path,
            records,
            json_lines=True,
        )
        receipt_binding = {
            key: binding[key]
            for key in (
                "source",
                "run_nonce",
                "attempt_nonce",
                "attempt_index",
                "env_step",
                "receipt_sha256",
                "file_sha256",
            )
        }
        fake._publish_task_memory(
            recipe_tag=task_spec.tag(config.public_seed),
            recipe_path=recipe_path,
            official_success_receipt=receipt_binding,
            source_artifacts_sha256=source_hashes,
        )
        _atomic_bytes(stage / "recipe.jsonl", recipe_path.read_bytes())
        _atomic_bytes(
            stage / "task_memory.md",
            (derivation / "memory" / f"{task_spec.task_name}.md").read_bytes(),
        )
        shutil.rmtree(derivation)
        recipe_sha256 = _sha256(stage / "recipe.jsonl")
        memory_sha256 = _sha256(stage / "task_memory.md")
        provenance = {
            "schema_version": 1,
            "candidate_only": True,
            "eligible_for_formal_eval": False,
            "review_required": True,
            "task_identity": {
                "task_name": task_spec.task_name,
                "task_index": task_spec.task_index,
                "activity_definition_id": task_spec.activity_definition_id,
                "activity_instance_id": config.candidate_instance_id,
            },
            "public_seed": config.public_seed,
            "source_tag": task_spec.tag(config.public_seed),
            "activity_instance_id": config.candidate_instance_id,
            "state": {
                "path": str(state_file),
                "sha256": config.candidate_state_sha256,
            },
            "planner": {
                "backend": "codex",
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
            },
            "raw_success": binding,
            "source_artifacts_sha256": source_hashes,
            "recipe_sha256": recipe_sha256,
            "memory_sha256": memory_sha256,
            "created_at": _utc_now(),
        }
        _atomic_json(stage / "provenance.json", provenance)
        amendment = {
            "schema_version": 1,
            "candidate_only": True,
            "eligible_for_formal_eval": False,
            "review_required": True,
            "publication_complete": True,
            "provenance_sha256": _sha256(stage / "provenance.json"),
            "recipe_sha256": recipe_sha256,
            "memory_sha256": memory_sha256,
            "created_at": _utc_now(),
        }
        _atomic_json(stage / "amendment.json", amendment)
        os.replace(stage, publication)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return amendment


def run_candidate_explore(
    config: ExploreConfig,
    *,
    candidate_state_file: Path,
    run_job: Callable[[ExploreConfig], dict[str, Any]] = run_explore_job,
) -> dict[str, Any]:
    """Run one finite candidate Job and seal its root-level evidence."""

    if not isinstance(config.resource_binding, DatasetResourceBinding):
        raise ValueError("candidate Explore requires pinned BEHAVIOR resources")
    verified_resources = verify_pinned_dataset_resources(config.resource_binding)
    if verified_resources.as_dict() != config.resource_binding.as_dict():
        raise ValueError("candidate BEHAVIOR resource binding changed before run")
    state_file = _validate_state_binding(config, candidate_state_file)
    outer_error: str | None = None
    try:
        manifest = run_job(config)
        if not isinstance(manifest, dict):
            raise TypeError("candidate runner returned no session manifest")
    except BaseException as error:
        outer_error = f"{type(error).__name__}: {error}"
        manifest = {
            "schema_version": 1,
            "campaign_kind": "candidate_instance_explore",
            "status": "outer_harness_failure",
            "task_success": None,
            "publication_complete": False,
            "protocol": {
                "task_index": get_task_spec(config.task_name).task_index,
                "task_name": config.task_name,
                "public_seed": config.public_seed,
            },
            "task_identity": {
                "task_name": config.task_name,
                "activity_definition_id": get_task_spec(
                    config.task_name
                ).activity_definition_id,
                "activity_instance_id": config.candidate_instance_id,
            },
            "native_binding": {
                "activity_instance_id": config.candidate_instance_id,
                "state_sha256": config.candidate_state_sha256,
            },
            "resource_source": config.resource_binding.as_dict(),
            "error": outer_error,
            "finished_at": _utc_now(),
            "attempts": [],
        }
    attempt_dir, binding = _seal_root_evidence(
        config,
        manifest,
        error=outer_error,
    )
    if binding is not None and attempt_dir is not None:
        amendment = _publish_candidate(
            config,
            state_file=state_file,
            attempt_dir=attempt_dir,
            binding=binding,
        )
        return {
            **manifest,
            "task_success": True,
            "publication_complete": amendment["publication_complete"],
            "candidate_publication": str(config.output_root / "candidate_publication"),
        }
    if manifest.get("task_success") is True:
        sealed_final = _read_json(config.output_root / "final_result.json")
        outer_error = (
            sealed_final.get("error")
            if isinstance(sealed_final, dict)
            and isinstance(sealed_final.get("error"), str)
            else "candidate claimed success without a valid raw-success receipt"
        )
    final = _read_json(config.output_root / "final_result.json")
    task_success = (
        False
        if isinstance(final, dict) and final.get("task_success") is False
        else None
    )
    return {
        **manifest,
        "task_success": task_success,
        "publication_complete": False,
        "error": (
            None
            if isinstance(manifest.get("terminal_failure"), dict)
            else outer_error
            or manifest.get("blocked_reason")
            or manifest.get("error")
            or "candidate did not produce verified raw official success"
        ),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one finite, candidate-only BEHAVIOR Explore job."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--activity-instance-id", type=int, required=True)
    parser.add_argument(
        "--task-name",
        choices=("turning_on_radio", "picking_up_trash"),
        default=TURNING_ON_RADIO_TASK_SPEC.task_name,
    )
    parser.add_argument("--public-seed", type=int, default=0)
    parser.add_argument("--candidate-state-file")
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo")
    parser.add_argument("--behavior-python")
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help="shared BEHAVIOR Pi0.5 checkpoint; task-specific SFTs are rejected",
    )
    parser.add_argument("--cuda-device", default="7")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--max-tool-calls", type=int, default=ATTEMPT_TOOL_CALLS)
    parser.add_argument(
        "--planner-timeout-s", type=int, default=CANDIDATE_PLANNER_TIMEOUT_S
    )
    parser.add_argument(
        "--attempt-timeout-s", type=int, default=CANDIDATE_ATTEMPT_TIMEOUT_S
    )
    parser.add_argument(
        "--max-wall-clock-s", type=int, default=CANDIDATE_MAX_WALL_CLOCK_S
    )
    parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--behavior-resource-revision",
        default=None,
        help=(
            "requested RLinf/RPent-memory revision; resolve and pin it once "
            "before reading BEHAVIOR memory or recipes"
        ),
    )
    parser.add_argument(
        "--behavior-resource-cache",
        default=None,
        help=(
            "versioned resource cache root (default: <repo-root>/resources/.snapshots)"
        ),
    )
    parser.add_argument(
        "--behavior-resource-offline",
        action="store_true",
        default=None,
        help="forbid network resolution/download; HF_HUB_OFFLINE=1 also enables it",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_state_file(args: argparse.Namespace, behavior_repo: Path) -> Path:
    if args.candidate_state_file:
        return Path(args.candidate_state_file).expanduser().resolve(strict=True)
    task_spec = get_task_spec(args.task_name)
    instance_dir = (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
        / "scenes"
        / task_spec.scene_model
        / "json"
        / task_spec.state_dir_name
    )
    matches = list(
        instance_dir.glob(f"*_{int(args.activity_instance_id)}_template-tro_state.json")
    )
    if len(matches) != 1:
        raise SystemExit(
            "expected exactly one candidate state file for instance "
            f"{args.activity_instance_id}, got {len(matches)}"
        )
    return matches[0].resolve(strict=True)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
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
    try:
        resource_binding = prepare_pinned_dataset_resources(
            "behavior",
            requested_revision=args.behavior_resource_revision,
            cache_root=(
                Path(args.behavior_resource_cache).expanduser().resolve()
                if args.behavior_resource_cache
                else repo_root / "resources" / ".snapshots"
            ),
            offline=args.behavior_resource_offline,
        )
        state_file = _resolve_state_file(args, behavior_repo)
        config = build_candidate_config(
            output_root=Path(args.output_root),
            repo_root=repo_root,
            python=Path(args.python),
            behavior_repo=behavior_repo,
            behavior_python=behavior_python,
            policy_checkpoint=Path(args.policy_checkpoint),
            candidate_instance_id=args.activity_instance_id,
            candidate_state_file=state_file,
            resource_binding=resource_binding,
            task_name=args.task_name,
            public_seed=args.public_seed,
            cuda_device=args.cuda_device,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_turns=args.max_turns,
            max_tool_calls=args.max_tool_calls,
            planner_timeout_s=args.planner_timeout_s,
            attempt_timeout_s=args.attempt_timeout_s,
            max_wall_clock_s=args.max_wall_clock_s,
            vla_ready_timeout_s=args.vla_ready_timeout_s,
            min_free_disk_gb=args.min_free_disk_gb,
            dashboard=args.dashboard,
            dashboard_host=args.dashboard_host,
            dashboard_port=args.dashboard_port,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    result = run_candidate_explore(config, candidate_state_file=state_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.dashboard and result.get("dashboard_url"):
        print(f"Dashboard remains available at {result.get('dashboard_url')}")
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass
    if result.get("task_success") is True:
        return 0
    return 2 if result.get("task_success") is None else 1


__all__ = [
    "CANDIDATE_ATTEMPT_TIMEOUT_S",
    "CANDIDATE_MAX_WALL_CLOCK_S",
    "CANDIDATE_PLANNER_TIMEOUT_S",
    "build_candidate_config",
    "main",
    "run_candidate_explore",
]
