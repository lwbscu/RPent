"""Runtime provider protocol for environment-specific CLI lifecycle."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rpent.tools.toolkit import Toolkit


@dataclass
class RuntimeHandle:
    """Resources created by an environment runtime for one agent run."""

    toolkit: Toolkit

    def close(self) -> None:
        """Release runtime resources after the cerebrum finishes."""
        self.toolkit.close()


class RuntimeProvider(Protocol):
    """Environment-specific lifecycle hooks used by the main CLI."""

    name: str

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Register environment-specific CLI arguments."""

    def validate_args(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        """Validate parsed CLI arguments, calling ``parser.error`` if needed."""

    def recipe_tag(self, args: argparse.Namespace) -> str:
        """Return the stable recipe/log tag for this run."""

    def dashboard_state(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
    ) -> Any:
        """Return an optional dashboard State for this run."""

    def prompt_vars(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        recipe_tag: str,
    ) -> dict[str, Any]:
        """Return variables used to render this environment's prompt bundle."""

    def start(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        dashboard: Any = None,
    ) -> RuntimeHandle:
        """Start environment resources and return a toolkit handle."""
