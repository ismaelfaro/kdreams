"""Tests for the kdream CLI using Click's test runner."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kdream.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestHelp:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "kdream" in result.output.lower()

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output

    def test_install_help(self, runner):
        result = runner.invoke(cli, ["install", "--help"])
        assert result.exit_code == 0

    def test_list_help(self, runner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0

    def test_generate_help(self, runner):
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--repo" in result.output

    def test_validate_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_packages_help(self, runner):
        result = runner.invoke(cli, ["packages", "--help"])
        assert result.exit_code == 0

    def test_cache_help(self, runner):
        result = runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0." in result.output  # semver present


class TestValidateCommand:
    def test_validate_valid_recipe(self, runner, sample_recipe_file):
        result = runner.invoke(cli, ["validate", str(sample_recipe_file), "--skip-verify"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_nonexistent_file(self, runner):
        result = runner.invoke(cli, ["validate", "/nonexistent/recipe.yaml"])
        assert result.exit_code != 0


class TestListCommand:
    def test_list_with_mocked_registry(self, runner):
        m = MagicMock()
        m.name = "stable-diffusion-xl-base"
        m.version = "1.0.0"
        m.tags = ["image-generation"]
        m.description = "Test"
        m.repo = "https://github.com/test/sdxl"

        with patch("kdream.list_recipes", return_value=[m]):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0

    def test_list_empty(self, runner):
        with patch("kdream.list_recipes", return_value=[]):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "No recipes" in result.output


class TestRunCommand:
    def test_run_with_mocked_kdream(self, runner):
        mock_result = MagicMock()
        mock_result.outputs = {"image": "/tmp/out.png"}
        mock_result.metadata = {"backend": "local", "duration_s": 1.0}

        with patch("kdream.run", return_value=mock_result):
            result = runner.invoke(cli, [
                "run", "stable-diffusion-xl-base",
                "--prompt", "test"
            ])
            assert result.exit_code == 0

    def test_run_error_exits_nonzero(self, runner):
        with patch("kdream.run", side_effect=Exception("Something broke")):
            result = runner.invoke(cli, ["run", "broken-recipe", "--prompt", "x"])
            assert result.exit_code != 0
            assert "Error" in result.output


class TestGenerateCommand:
    def test_generate_requires_repo(self, runner):
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0

    def test_generate_z_image(self, runner, tmp_path):
        """generate command works for Z-Image repo (mocked)."""
        mock_recipe = MagicMock()
        mock_recipe.metadata.name = "z-image"
        out = tmp_path / "z-image.yaml"

        with patch("kdream.generate_recipe", return_value=mock_recipe):
            result = runner.invoke(cli, [
                "generate",
                "--repo", "https://github.com/Tongyi-MAI/Z-Image",
                "--output", str(out),
            ])
            assert result.exit_code == 0

    def test_generate_corridorkey(self, runner, tmp_path):
        """generate command works for CorridorKey repo (mocked)."""
        mock_recipe = MagicMock()
        mock_recipe.metadata.name = "corridorkey"
        out = tmp_path / "corridorkey.yaml"

        with patch("kdream.generate_recipe", return_value=mock_recipe):
            result = runner.invoke(cli, [
                "generate",
                "--repo", "https://github.com/nikopueringer/CorridorKey",
                "--output", str(out),
            ])
            assert result.exit_code == 0


class TestCacheCommand:
    def test_cache_info(self, runner):
        result = runner.invoke(cli, ["cache", "info"])
        assert result.exit_code == 0

    def test_cache_clear_aborts_on_no(self, runner):
        result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        assert result.exit_code != 0 or "Aborted" in result.output


class TestPackagesCommand:
    def test_packages_empty(self, runner):
        with patch("kdream.list_installed", return_value=[]):
            result = runner.invoke(cli, ["packages"])
            assert result.exit_code == 0
            assert "No packages" in result.output

    def test_packages_with_entries(self, runner, tmp_path):
        from kdream.core.runner import PackageInfo
        pkg = PackageInfo(
            recipe_name="test-model",
            path=tmp_path / "test-model",
            ready=True,
            venv_path=tmp_path / "test-model" / "venv",
            repo_path=tmp_path / "test-model" / "repo",
            models_path=tmp_path / "test-model" / "models",
        )
        with patch("kdream.list_installed", return_value=[pkg]):
            result = runner.invoke(cli, ["packages"])
            assert result.exit_code == 0
            assert "test-model" in result.output
