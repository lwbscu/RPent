"""Structural regression tests for the BEHAVIOR EnvSpec migration.

BEHAVIOR mirrors LIBERO's plugin layout while retaining its own robot, task,
success, and publication semantics.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
PROMPT_ROOT = BEHAVIOR_ROOT / "prompts"
GUIDE_ROOT = BEHAVIOR_ROOT / "guides"


def _production_python() -> str:
    roots = (REPO_ROOT / "robots" / "behavior", REPO_ROOT / "rpent")
    return "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in sorted(root.rglob("*.py"))
    )


def _behavior_prompt_and_guide_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for root in (PROMPT_ROOT, GUIDE_ROOT)
        for path in sorted(root.iterdir())
        if path.suffix in {".py", ".md"}
    )


def test_behavior_plugin_matches_libero_prompt_and_guide_layout() -> None:
    assert {path.name for path in PROMPT_ROOT.iterdir() if path.is_file()} == {
        "__init__.py",
        "system.py",
        "user.py",
    }
    assert {path.name for path in GUIDE_ROOT.iterdir() if path.is_file()} == {
        "R1Pro robot prior.md",
        "env_calibration.md",
        "pro_hybrid_guide.md",
        "strict_hybrid_guide.md",
    }


@pytest.mark.parametrize(
    "removed",
    (
        BEHAVIOR_ROOT / "runtime_provider.py",
        BEHAVIOR_ROOT / "README.md",
        PROMPT_ROOT / "agent_task_prompt_explore.md",
        PROMPT_ROOT / "agent_task_prompt_perception_fresh.md",
        REPO_ROOT / "rpent" / "envs" / "runtime.py",
    ),
)
def test_removed_behavior_prompt_and_runtime_sources_do_not_return(
    removed: Path,
) -> None:
    assert not removed.exists()


def test_production_imports_use_envspec_instead_of_runtime_provider() -> None:
    source = _production_python()
    assert "robots.behavior.runtime_provider" not in source
    assert "get_runtime_provider" not in source


@pytest.mark.parametrize(
    "script_name",
    ("run_behavior_serial_explore.py", "run_behavior_serial_eval.py"),
)
def test_behavior_serial_wrappers_are_runnable_outside_repo_root(
    script_name: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--task-name" in completed.stdout


@pytest.mark.parametrize(
    "forbidden",
    (
        "libero-pro",
        "libero_pro",
        "libero_terminated",
        "sam3",
        "franka",
        "7d action",
        "5-step",
        "read_text_file",
        "write_text_file",
        "list_dir",
    ),
)
def test_behavior_prompts_and_guides_do_not_import_libero_execution_semantics(
    forbidden: str,
) -> None:
    source = _behavior_prompt_and_guide_source().lower()
    assert forbidden not in source
