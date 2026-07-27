"""Static env-extension descriptor.

Lives in :mod:`rpent.envs` alongside
:class:`~rpent.envs.prompt_bundle.PromptBundle` so envs
and planners can both import it without pulling in
:mod:`rpent.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`rpent.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries the env identity, prompt bundle, required setup hooks,
resource policy, and optional planner/finalizer hooks that keep
``rpent/cli/main.py`` env-agnostic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rpent.envs.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from robots.behavior.dashboard_state import State
    from rpent.planner.base import Planner, PlannerResult
    from rpent.tools.toolkit import Toolkit

ResourcePolicy = Literal["sync_remote", "frozen_local"]


class RuntimeResource(Protocol):
    """One owned runtime resource released by the shared CLI."""

    def stop(self) -> None:
        """Release this resource. Implementations must make this idempotent."""


class RunPlannerHook(Protocol):
    """Environment override for the planner execution phase."""

    def __call__(
        self,
        *,
        planner: "Planner",
        system_prompt: str,
        user_message: str,
        toolkit: "Toolkit",
        max_turns: int,
        input_queue: Any,
        args: argparse.Namespace,
        run_config: "RunConfig",
        runtime_resources: Sequence[RuntimeResource],
    ) -> "PlannerResult":
        """Run the environment-specific planner loop."""


class FinalizeRunHook(Protocol):
    """Environment override for artifact finalization."""

    def __call__(self, outcome: "RunOutcome") -> None:
        """Finalize artifacts after toolkit and runtime cleanup have run."""


class PrepareResourcesHook(Protocol):
    """Environment override for strict resource preparation before config parse."""

    def __call__(self, args: argparse.Namespace) -> Any | None:
        """Prepare resources and return a binding exposed on ``args``."""


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`EnvSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    dashboard_state: "State | None"
    task_desc: dict[str, Any]
    resource_policy: ResourcePolicy | None = None


@dataclass
class RunOutcome:
    """Complete run state passed to an environment finalizer after cleanup."""

    args: argparse.Namespace
    run_config: RunConfig
    toolkit: "Toolkit | None"
    runtime_resources: tuple[RuntimeResource, ...]
    finish_result: dict[str, Any] | None
    messages: list[dict[str, Any]]
    stats: dict[str, Any]
    error: str | None
    elapsed_s: float
    transcript_path: Path
    recipe_path: str | None
    cleanup: dict[str, Any]
    task_desc: dict[str, Any]
    prompt_vars: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for RPent."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path],
        tuple[list[RuntimeResource], dict[str, Any]],
    ]
    resource_policy: ResourcePolicy = "sync_remote"
    prepare_resources: PrepareResourcesHook | None = None
    run_planner: RunPlannerHook | None = None
    finalize_run: FinalizeRunHook | None = None
