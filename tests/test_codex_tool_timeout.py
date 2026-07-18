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

