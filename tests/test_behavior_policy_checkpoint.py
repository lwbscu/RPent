from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robots.behavior import policy_checkpoint, vla_server
from robots.behavior.policy_checkpoint import (
    CheckpointFileRequirement,
    PolicyCheckpointError,
    PolicyCheckpointProfile,
)
from robots.behavior.vla_client import BehaviorVLAClient


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_profile(root: Path) -> PolicyCheckpointProfile:
    files = {
        "model.safetensors": b"general behavior policy",
        "config.json": b'{"action_horizon":32}\n',
        "assets/behavior-1k/2025-challenge-demos/norm_stats.json": (
            b'{"norm":"general"}\n'
        ),
    }
    requirements = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        requirements.append(
            CheckpointFileRequirement(
                relative_path=relative,
                size_bytes=len(content),
                sha256=_sha256(path),
            )
        )
    return PolicyCheckpointProfile(
        profile_id="pi05-b1kpt50-cs32",
        path=root,
        files=tuple(requirements),
    )


def test_shared_profile_has_the_verified_general_checkpoint_identity():
    assert policy_checkpoint.SHARED_POLICY_PROFILE_ID == "pi05-b1kpt50-cs32"
    assert policy_checkpoint.SHARED_POLICY_CHECKPOINT_PATH == Path(
        "/home/ubuntu/lwb/Models/openpi_comet_pytorch/pi05-b1kpt50-cs32"
    )
    requirements = {
        item.relative_path: (item.size_bytes, item.sha256)
        for item in policy_checkpoint.SHARED_POLICY_PROFILE.files
    }
    assert requirements == {
        "model.safetensors": (
            7_233_650_408,
            "7e257666d835f6af701de493676a6c86a0421b2efc737a0f911d782b7a09f635",
        ),
        "config.json": (
            149,
            "a4ae208203adfdd64c5fdbd4b0dc257e4ebbc82e464cb146dd0377051b25fc0a",
        ),
        "assets/behavior-1k/2025-challenge-demos/norm_stats.json": (
            6_368,
            "d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929",
        ),
    }


def test_validator_returns_a_stable_public_binding(monkeypatch, tmp_path):
    profile = _fake_profile(tmp_path / "checkpoint")
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)

    first = policy_checkpoint.validate_policy_checkpoint(profile.path)
    second = policy_checkpoint.validate_policy_checkpoint(profile.path)

    assert first == second
    payload = first.as_dict()
    assert payload["profile_id"] == "pi05-b1kpt50-cs32"
    assert payload["resolved_path"] == str(profile.path.resolve())
    unsigned = dict(payload)
    del unsigned["binding_sha256"]
    assert (
        payload["binding_sha256"]
        == hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_validator_rejects_an_identical_checkpoint_at_another_path(
    monkeypatch,
    tmp_path,
):
    profile = _fake_profile(tmp_path / "general")
    alternate = tmp_path / "task-specific-sft"
    alternate_profile = _fake_profile(alternate)
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)

    with pytest.raises(PolicyCheckpointError, match="requires the shared"):
        policy_checkpoint.validate_policy_checkpoint(alternate_profile.path)


def test_validator_rejects_changed_checkpoint_contents(monkeypatch, tmp_path):
    profile = _fake_profile(tmp_path / "checkpoint")
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)
    (profile.path / "model.safetensors").write_bytes(b"task-specific SFT")

    with pytest.raises(
        PolicyCheckpointError,
        match="size mismatch|SHA256 mismatch",
    ):
        policy_checkpoint.validate_policy_checkpoint(profile.path)


def test_binding_comparison_rejects_wrong_vla_identity(monkeypatch, tmp_path):
    profile = _fake_profile(tmp_path / "checkpoint")
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)
    expected = policy_checkpoint.validate_policy_checkpoint(profile.path)
    wrong = expected.as_dict()
    wrong["profile_id"] = "pi05-turning_on_radio-sft"

    with pytest.raises(PolicyCheckpointError, match="does not match"):
        policy_checkpoint.assert_matching_policy_checkpoint_binding(wrong, expected)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HTTPClient:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, **kwargs):
        del url, kwargs
        return _Response(self.payload)


def test_vla_client_checks_health_checkpoint_binding(monkeypatch, tmp_path):
    profile = _fake_profile(tmp_path / "checkpoint")
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)
    expected = policy_checkpoint.validate_policy_checkpoint(profile.path)
    client = BehaviorVLAClient.__new__(BehaviorVLAClient)
    client._base_url = "http://127.0.0.1:1"
    client._client = _HTTPClient(
        {
            "status": "ok",
            "checkpoint_binding": expected.as_dict(),
        }
    )

    payload = client.healthz(expected_checkpoint_binding=expected)
    assert payload["checkpoint_binding"] == expected.as_dict()

    client._client.payload["checkpoint_binding"]["binding_sha256"] = "0" * 64
    with pytest.raises(PolicyCheckpointError, match="does not match"):
        client.healthz(expected_checkpoint_binding=expected)


def test_vla_health_exposes_loaded_checkpoint_binding(monkeypatch, tmp_path):
    profile = _fake_profile(tmp_path / "checkpoint")
    monkeypatch.setattr(policy_checkpoint, "SHARED_POLICY_PROFILE", profile)
    expected = policy_checkpoint.validate_policy_checkpoint(profile.path)
    monkeypatch.setattr(vla_server, "_MODEL", object())
    monkeypatch.setattr(
        vla_server,
        "_MODEL_META",
        {
            "status": "ok",
            "checkpoint_binding": expected.as_dict(),
        },
    )
    app = vla_server.build_app()
    health = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/healthz"
    )

    payload = health()
    assert payload["checkpoint_binding"] == expected.as_dict()
