"""Small, non-production BEHAVIOR resource bindings used by unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    DatasetResourceFile,
)

FIXTURE_DATASET_REPO = "fixture/RPent-memory"
FIXTURE_REVISION = "a" * 40
FIXTURE_RESOURCES = (
    Path(__file__).resolve().parent / "fixtures" / "behavior_resources" / "behavior"
)


def fixture_resource_binding(
    root: Path = FIXTURE_RESOURCES,
    *,
    requested_revision: str = FIXTURE_REVISION,
    resolved_revision: str = FIXTURE_REVISION,
    offline: bool = True,
) -> DatasetResourceBinding:
    """Return the binding declared by the checked-in minimal fixture."""

    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetResourceBinding(
        dataset_repo=FIXTURE_DATASET_REPO,
        repo_type="dataset",
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        subtree="behavior",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        files=tuple(
            DatasetResourceFile(
                path=item["path"],
                size_bytes=item["size_bytes"],
                sha256=item["sha256"],
            )
            for item in payload["files"]
        ),
        offline=offline,
        root=root.resolve(),
    )
