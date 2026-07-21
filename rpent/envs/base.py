"""Env registry: maps env name to environment factories.

Env implementations live in the top-level ``robots/`` directory (a sibling of
the ``rpent`` package); an env is resolved by importing ``robots.<name>``. The
``EnvSpec`` / ``PromptBundle`` dataclasses themselves live in :mod:`rpent.envs`
so cerebrums and envs share the same contract types without crossing module
layers.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.envs.runtime import RuntimeProvider
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root

# Env packages live under ``<repo>/robots/``, which is not part of the installed
# ``rpent`` distribution. Ensure the repo root is importable so ``robots.<name>``
# resolves regardless of the process's current working directory.
_REPO_ROOT = str(get_repo_root())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_env(name: str) -> Any:
    """Import ``robots.<name>`` lazily and return the module."""
    if not name:
        raise ValueError("env name must be non-empty")
    env_name = name.lower()
    try:
        return importlib.import_module(f"robots.{env_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown env: {env_name!r}") from e


def get_env_spec(name: str) -> EnvSpec:
    """Load an environment's declarative specification."""
    return _resolve_env(name).get_env_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """Build the env toolkit (common tools + env-specific tools)."""
    return _resolve_env(name).get_toolkit(**kwargs)


def get_runtime_provider(name: str) -> RuntimeProvider:
    """Build the env runtime provider used by the main CLI."""
    env_module = _resolve_env(name)
    try:
        factory = env_module.get_runtime_provider
    except AttributeError as e:
        env_name = name.lower()
        raise ValueError(f"env {env_name!r} does not provide a runtime provider") from e
    return factory()
