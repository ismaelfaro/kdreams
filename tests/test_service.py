"""Tests for kdream service mode: MCP server, ngrok tunnel, and remote CLI."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kdream.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# ── serve command ─────────────────────────────────────────────────────────────


class TestServeCommand:
    def test_serve_help(self, runner):
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "--ngrok" in result.output
        assert "--port" in result.output

    def test_serve_missing_deps_exits_nonzero(self, runner):
        with patch.dict("sys.modules", {"mcp": None, "mcp.server": None,
                                        "mcp.server.fastmcp": None}):
            with patch("kdream.cli._remote_client",
                       side_effect=SystemExit(1)):
                pass  # covered by import-error path below

        # Simulate ImportError from create_mcp_server
        with patch("kdream.service.mcp_server.create_mcp_server",
                   side_effect=ImportError("no mcp")):
            # patch the import inside the command
            with patch.dict("sys.modules", {}):
                import builtins
                real_import = builtins.__import__

                def fake_import(name, *args, **kwargs):
                    if name == "kdream.service.mcp_server":
                        raise ImportError("no mcp")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    result = runner.invoke(cli, ["serve", "--transport", "stdio"])
                    assert result.exit_code == 1
                    assert "MCP dependencies not installed" in result.output

    def test_serve_stdio_calls_mcp_run(self, runner):
        mock_mcp = MagicMock()
        with patch("kdream.service.mcp_server.create_mcp_server",
                   return_value=mock_mcp):
            result = runner.invoke(cli, ["serve", "--transport", "stdio"])
        mock_mcp.run.assert_called_once_with(transport="stdio")
        assert result.exit_code == 0

    def test_serve_http_calls_streamable_http(self, runner):
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt  # simulate Ctrl-C
        with patch("kdream.service.mcp_server.create_mcp_server",
                   return_value=mock_mcp):
            result = runner.invoke(cli, ["serve", "--transport", "http"])
        mock_mcp.run.assert_called_once_with(transport="streamable-http")
        assert result.exit_code == 0
        assert "Server stopped" in result.output

    def test_serve_ngrok_starts_and_stops_tunnel(self, runner):
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt

        mock_tunnel = MagicMock()
        mock_tunnel.start.return_value = "https://abc123.ngrok.io"

        with patch("kdream.service.mcp_server.create_mcp_server",
                   return_value=mock_mcp):
            with patch("kdream.service.ngrok_tunnel.NgrokTunnel",
                       return_value=mock_tunnel):
                result = runner.invoke(cli, [
                    "serve", "--transport", "http", "--ngrok",
                ])

        mock_tunnel.start.assert_called_once()
        mock_tunnel.stop.assert_called_once()
        assert "https://abc123.ngrok.io" in result.output

    def test_serve_http_default_port_and_host(self, runner):
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt

        with patch("kdream.service.mcp_server.create_mcp_server",
                   return_value=mock_mcp) as mock_factory:
            runner.invoke(cli, ["serve", "--transport", "http"])

        mock_factory.assert_called_once_with(host="127.0.0.1", port=8765)

    def test_serve_custom_port(self, runner):
        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = KeyboardInterrupt

        with patch("kdream.service.mcp_server.create_mcp_server",
                   return_value=mock_mcp) as mock_factory:
            runner.invoke(cli, ["serve", "--transport", "http", "--port", "9000"])

        mock_factory.assert_called_once_with(host="127.0.0.1", port=9000)


# ── create_mcp_server ─────────────────────────────────────────────────────────


class TestCreateMcpServer:
    def test_returns_fastmcp_instance(self):
        from mcp.server.fastmcp import FastMCP
        from kdream.service.mcp_server import create_mcp_server

        mcp = create_mcp_server()
        assert isinstance(mcp, FastMCP)

    def test_all_tools_registered(self):
        from kdream.service.mcp_server import create_mcp_server

        mcp = create_mcp_server()
        # FastMCP stores tools in _tool_manager
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        expected = {
            "run_recipe", "install_recipe", "list_recipes",
            "generate_recipe", "validate_recipe", "list_installed",
        }
        assert expected == tool_names

    def test_run_recipe_tool_raises_tool_error_on_kdream_error(self):
        from mcp.server.fastmcp.exceptions import ToolError
        from kdream.service.mcp_server import create_mcp_server
        from kdream.exceptions import KdreamError

        mcp = create_mcp_server()
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "run_recipe")

        with patch("kdream.run", side_effect=KdreamError("boom")):
            with pytest.raises(ToolError, match="boom"):
                import asyncio
                asyncio.run(tool.run({"recipe": "test-recipe"}))

    def test_list_recipes_tool_returns_list(self):
        from kdream.service.mcp_server import create_mcp_server

        mcp = create_mcp_server()
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "list_recipes")

        mock_recipe = MagicMock()
        mock_recipe.name = "stable-diffusion-xl-base"
        mock_recipe.version = "1.0.0"
        mock_recipe.description = "Test"
        mock_recipe.tags = ["image-generation"]
        mock_recipe.repo = "https://github.com/test/repo"

        with patch("kdream.list_recipes", return_value=[mock_recipe]):
            import asyncio
            result = asyncio.run(tool.run({}))

        assert isinstance(result, list)
        assert result[0]["name"] == "stable-diffusion-xl-base"

    def test_validate_recipe_tool_returns_valid_dict(self, tmp_path):
        from kdream.service.mcp_server import create_mcp_server

        mcp = create_mcp_server()
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "validate_recipe")

        mock_recipe = MagicMock()
        mock_recipe.metadata.name = "test-model"
        mock_recipe.metadata.version = "1.0.0"
        mock_recipe.inputs = {"prompt": MagicMock()}
        mock_recipe.models = [MagicMock()]
        mock_recipe.outputs = [MagicMock()]

        with patch("kdream.load_recipe", return_value=mock_recipe):
            with patch("kdream.validate_recipe", return_value=[]):
                import asyncio
                result = asyncio.run(tool.run({"recipe_file": "/fake/recipe.yaml"}))

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["name"] == "test-model"

    def test_list_installed_tool_returns_list(self, tmp_path):
        from kdream.service.mcp_server import create_mcp_server
        from kdream.core.runner import PackageInfo

        mcp = create_mcp_server()
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "list_installed")

        pkg = PackageInfo(
            recipe_name="my-model",
            path=tmp_path / "my-model",
            ready=True,
            venv_path=tmp_path / "my-model" / "venv",
            repo_path=tmp_path / "my-model" / "repo",
            models_path=tmp_path / "my-model" / "models",
        )

        with patch("kdream.list_installed", return_value=[pkg]):
            import asyncio
            result = asyncio.run(tool.run({}))

        assert isinstance(result, list)
        assert result[0]["recipe_name"] == "my-model"
        assert result[0]["ready"] is True


# ── NgrokTunnel ───────────────────────────────────────────────────────────────


class TestNgrokTunnel:
    def _make_tunnel_mock(self, url="http://abc123.ngrok.io"):
        m = MagicMock()
        m.public_url = url
        return m

    def test_start_returns_https_url(self):
        from kdream.service.ngrok_tunnel import NgrokTunnel

        mock_tunnel_obj = self._make_tunnel_mock("http://abc123.ngrok.io")
        with patch("pyngrok.ngrok.connect", return_value=mock_tunnel_obj):
            tunnel = NgrokTunnel(port=8765)
            url = tunnel.start()

        assert url == "https://abc123.ngrok.io"

    def test_http_url_normalised_to_https(self):
        from kdream.service.ngrok_tunnel import NgrokTunnel

        mock_tunnel_obj = self._make_tunnel_mock("http://xyz.ngrok.io")
        with patch("pyngrok.ngrok.connect", return_value=mock_tunnel_obj):
            tunnel = NgrokTunnel(port=8765)
            url = tunnel.start()

        assert url == "https://xyz.ngrok.io"

    def test_stop_clears_state(self):
        from kdream.service.ngrok_tunnel import NgrokTunnel

        mock_tunnel_obj = self._make_tunnel_mock()
        with patch("pyngrok.ngrok.connect", return_value=mock_tunnel_obj):
            with patch("pyngrok.ngrok.disconnect"):
                tunnel = NgrokTunnel(port=8765)
                tunnel.start()
                tunnel.stop()

        assert tunnel.public_url is None
        assert tunnel._tunnel is None

    def test_context_manager(self):
        from kdream.service.ngrok_tunnel import NgrokTunnel

        mock_tunnel_obj = self._make_tunnel_mock()
        with patch("pyngrok.ngrok.connect", return_value=mock_tunnel_obj):
            with patch("pyngrok.ngrok.disconnect"):
                with NgrokTunnel(port=8765) as t:
                    assert t.public_url is not None
                assert t.public_url is None

    def test_missing_pyngrok_raises_runtime_error(self):
        from kdream.service.ngrok_tunnel import NgrokTunnel
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyngrok":
                raise ImportError("no pyngrok")
            return real_import(name, *args, **kwargs)

        tunnel = NgrokTunnel(port=8765)
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="pyngrok is required"):
                tunnel.start()


# ── remote CLI commands ───────────────────────────────────────────────────────


class TestRemoteCommand:
    def test_remote_help(self, runner):
        result = runner.invoke(cli, ["remote", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "list" in result.output
        assert "packages" in result.output

    def test_remote_run_help(self, runner):
        result = runner.invoke(cli, ["remote", "run", "--help"])
        assert result.exit_code == 0
        assert "--url" in result.output
        assert "--prompt" in result.output

    def test_remote_run_success(self, runner):
        mock_result = {
            "success": True,
            "outputs": {"image": "/remote/outputs/out.png"},
            "metadata": {"backend": "local", "duration_s": 2.5},
            "error": None,
        }
        with patch("kdream.service.mcp_client.call_tool", return_value=mock_result):
            result = runner.invoke(cli, [
                "remote", "run", "stable-diffusion-xl-base",
                "--url", "http://host:8765/mcp",
                "--prompt", "red panda hacker",
            ])
        assert result.exit_code == 0
        assert "Remote inference complete" in result.output
        assert "/remote/outputs/out.png" in result.output

    def test_remote_run_remote_error_exits_nonzero(self, runner):
        mock_result = {
            "success": False,
            "outputs": {},
            "metadata": {},
            "error": "Model not found",
        }
        with patch("kdream.service.mcp_client.call_tool", return_value=mock_result):
            result = runner.invoke(cli, [
                "remote", "run", "bad-recipe",
                "--url", "http://host:8765/mcp",
            ])
        assert result.exit_code != 0
        assert "Remote error" in result.output

    def test_remote_run_connection_error_exits_nonzero(self, runner):
        with patch("kdream.service.mcp_client.call_tool",
                   side_effect=Exception("Connection refused")):
            result = runner.invoke(cli, [
                "remote", "run", "stable-diffusion-xl-base",
                "--url", "http://unreachable:9999/mcp",
                "--prompt", "test",
            ])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_remote_list_shows_table(self, runner):
        mock_recipes = [
            {
                "name": "stable-diffusion-xl-base",
                "version": "1.0.0",
                "description": "Image generation",
                "tags": ["image-generation"],
                "repo": "https://github.com/test/sdxl",
            }
        ]
        with patch("kdream.service.mcp_client.call_tool", return_value=mock_recipes):
            result = runner.invoke(cli, [
                "remote", "list", "--url", "http://host:8765/mcp",
            ])
        assert result.exit_code == 0
        assert "stable-diffusion-xl-base" in result.output

    def test_remote_list_empty(self, runner):
        with patch("kdream.service.mcp_client.call_tool", return_value=[]):
            result = runner.invoke(cli, [
                "remote", "list", "--url", "http://host:8765/mcp",
            ])
        assert result.exit_code == 0
        assert "No recipes" in result.output

    def test_remote_packages_shows_table(self, runner):
        mock_pkgs = [
            {"recipe_name": "llama-3-8b-instruct", "path": "/remote/cache/llama", "ready": True}
        ]
        with patch("kdream.service.mcp_client.call_tool", return_value=mock_pkgs):
            result = runner.invoke(cli, [
                "remote", "packages", "--url", "http://host:8765/mcp",
            ])
        assert result.exit_code == 0
        assert "llama-3-8b-instruct" in result.output

    def test_remote_packages_empty(self, runner):
        with patch("kdream.service.mcp_client.call_tool", return_value=[]):
            result = runner.invoke(cli, [
                "remote", "packages", "--url", "http://host:8765/mcp",
            ])
        assert result.exit_code == 0
        assert "No packages" in result.output

    def test_remote_run_args_forwarded(self, runner):
        mock_result = {
            "success": True, "outputs": {}, "metadata": {}, "error": None,
        }
        captured = {}

        def capture(url, tool, args):
            captured.update(args)
            return mock_result

        with patch("kdream.service.mcp_client.call_tool", side_effect=capture):
            runner.invoke(cli, [
                "remote", "run", "some-recipe",
                "--url", "http://host:8765/mcp",
                "--prompt", "hello",
                "--steps", "20",
                "--seed", "42",
            ])

        assert captured.get("recipe") == "some-recipe"
        assert captured.get("prompt") == "hello"
        assert captured.get("steps") == 20
        assert captured.get("seed") == 42


# ── mcp_client ────────────────────────────────────────────────────────────────


class TestMcpClient:
    def _make_session_mock(self, tool_result):
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=tool_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    def _make_transport_mock(self):
        mock_transport = AsyncMock()
        mock_transport.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), None))
        mock_transport.__aexit__ = AsyncMock(return_value=False)
        return mock_transport

    def test_call_tool_parses_json_response(self):
        """call_tool returns parsed JSON from text content."""
        import asyncio
        import kdream.service.mcp_client as client_mod
        from kdream.service.mcp_client import _call_tool

        payload = {"success": True, "outputs": {"image": "/out.png"}}
        mock_content = MagicMock()
        mock_content.text = json.dumps(payload)
        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session = self._make_session_mock(mock_result)
        mock_transport = self._make_transport_mock()

        with patch.object(client_mod, "ClientSession", return_value=mock_session):
            with patch.object(client_mod, "streamablehttp_client", return_value=mock_transport):
                result = asyncio.run(_call_tool("http://host/mcp", "run_recipe", {}))

        assert result["success"] is True
        assert result["outputs"]["image"] == "/out.png"

    def test_call_tool_empty_content_returns_empty_dict(self):
        import asyncio
        import kdream.service.mcp_client as client_mod
        from kdream.service.mcp_client import _call_tool

        mock_result = MagicMock()
        mock_result.content = []

        mock_session = self._make_session_mock(mock_result)
        mock_transport = self._make_transport_mock()

        with patch.object(client_mod, "ClientSession", return_value=mock_session):
            with patch.object(client_mod, "streamablehttp_client", return_value=mock_transport):
                result = asyncio.run(_call_tool("http://host/mcp", "list_recipes", {}))

        assert result == {}
