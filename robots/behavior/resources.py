"""Strict, task-neutral resource preparation for BEHAVIOR runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    ResourcePreparationError,
    load_dataset_resource_binding,
    prepare_local_dataset_resources,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
)

BEHAVIOR_RESOURCE_SUBTREE = "behavior"


def _install_binding(
    args: argparse.Namespace,
    binding: DatasetResourceBinding,
) -> DatasetResourceBinding:
    """Expose one verified binding to config loaders and run artifacts."""

    verified = verify_pinned_dataset_resources(binding)
    if verified.subtree != BEHAVIOR_RESOURCE_SUBTREE:
        raise ResourcePreparationError(
            "BEHAVIOR resource binding must select the behavior subtree"
        )
    root = verified.root.expanduser().resolve(strict=True)
    args._behavior_resource_root = root
    args._behavior_resource_source = verified.as_dict()
    args.prepared_resources = verified
    return verified


def _load_serial_child_binding(
    args: argparse.Namespace,
    *,
    root_value: str,
    source_file_value: str,
) -> DatasetResourceBinding:
    """Revalidate a parent's pinned snapshot without network access."""

    binding = load_dataset_resource_binding(Path(source_file_value).expanduser())
    requested_root = Path(root_value).expanduser().resolve(strict=True)
    bound_root = binding.root.expanduser().resolve(strict=True)
    if requested_root != bound_root:
        raise ResourcePreparationError(
            "serial child BEHAVIOR resource root does not match its source binding"
        )
    return _install_binding(args, binding)


def prepare_behavior_resources(
    args: argparse.Namespace,
) -> DatasetResourceBinding:
    """Prepare or revalidate exactly one immutable BEHAVIOR resource snapshot."""

    root_value = getattr(args, "behavior_resource_root", None)
    source_file_value = getattr(args, "behavior_resource_source_file", None)
    if (root_value is None) != (source_file_value is None):
        raise ResourcePreparationError(
            "--behavior-resource-root and --behavior-resource-source-file "
            "must be provided together"
        )
    if root_value is not None:
        return _load_serial_child_binding(
            args,
            root_value=str(root_value),
            source_file_value=str(source_file_value),
        )

    cache_value = getattr(args, "behavior_resource_cache", None)
    cache_root = (
        Path(cache_value).expanduser()
        if cache_value is not None and str(cache_value).strip()
        else None
    )
    offline_value = getattr(args, "behavior_resource_offline", None)
    offline = True if offline_value is True else None
    local_value = getattr(args, "behavior_resource_local", None)
    if local_value is not None and str(local_value).strip():
        revision_value = getattr(args, "behavior_resource_revision", None)
        if revision_value is not None and str(revision_value).strip():
            raise ResourcePreparationError(
                "--behavior-resource-local and --behavior-resource-revision "
                "are mutually exclusive"
            )
        binding = prepare_local_dataset_resources(
            BEHAVIOR_RESOURCE_SUBTREE,
            source_root=Path(str(local_value)).expanduser(),
            cache_root=cache_root,
        )
        return _install_binding(args, binding)
    binding = prepare_pinned_dataset_resources(
        BEHAVIOR_RESOURCE_SUBTREE,
        requested_revision=getattr(args, "behavior_resource_revision", None),
        cache_root=cache_root,
        offline=offline,
    )
    return _install_binding(args, binding)


__all__ = [
    "BEHAVIOR_RESOURCE_SUBTREE",
    "prepare_behavior_resources",
]
