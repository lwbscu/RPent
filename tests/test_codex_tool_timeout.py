import rpent.cerebrum.codex as codex_module
from rpent.cerebrum.codex import CodexCerebrum, _codex_mcp_config_overrides


def test_codex_mcp_config_contains_tool_timeout():
    overrides = _codex_mcp_config_overrides(
        mcp_url="http://127.0.0.1:9911/mcp",
        base_url=None,
        tool_timeout_sec=777,
    )

    assert "mcp_servers.rpent.url=\"http://127.0.0.1:9911/mcp\"" in overrides
    assert "mcp_servers.rpent.tool_timeout_sec=777" in overrides


def test_codex_cerebrum_uses_wall_clock_timeout_as_tool_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        codex_module.openai_codex,
        "CodexConfig",
        lambda **kwargs: kwargs,
    )
    cerebrum = CodexCerebrum(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        timeout_s=654,
    )

    config = cerebrum._build_config("http://127.0.0.1:9911/mcp")

    assert "mcp_servers.rpent.tool_timeout_sec=654" in config["config_overrides"]


def test_codex_tool_only_config_disables_non_mcp_capabilities(monkeypatch, tmp_path):
    monkeypatch.setattr(
        codex_module.openai_codex,
        "CodexConfig",
        lambda **kwargs: kwargs,
    )
    cerebrum = CodexCerebrum(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        timeout_s=654,
        tool_only=True,
    )

    config = cerebrum._build_config("http://127.0.0.1:9911/mcp")
    overrides = set(config["config_overrides"])

    assert "mcp_servers={}" in overrides
    assert 'mcp_servers.rpent.default_tools_approval_mode="approve"' in overrides
    assert "features.shell_tool=false" in overrides
    assert "features.unified_exec=false" in overrides
    assert "features.apps=false" in overrides
    assert "features.multi_agent=false" in overrides
    assert "features.remote_plugin=false" in overrides
    assert 'web_search="disabled"' in overrides
