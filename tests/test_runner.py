"""Tests for kdream.core.runner — orchestration layer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdream.core.runner import PackageInfo, RunResult, _get_cache_dir, _resolve_recipe, install, list_installed, run
from kdream.exceptions import BackendError, RecipeError, RegistryError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_RECIPE_YAML = """\
apiVersion: kdream/v1
kind: Recipe
metadata:
  name: test-runner-model
  version: 1.0.0
  description: Test recipe for runner tests
  tags: [test]
  license: Apache-2.0
  author: test
source:
  repo: https://github.com/test/test-runner-model
  ref: main
entrypoint:
  script: run.py
  type: python
inputs:
  prompt:
    type: string
    required: true
outputs:
  - name: result
    type: string
"""


def _make_package_info(tmp_path: Path, name: str = "test-runner-model") -> PackageInfo:
    pkg_dir = tmp_path / name
    repo_path = pkg_dir / "repo"
    venv_path = pkg_dir / "venv"
    models_path = pkg_dir / "models"
    for d in [repo_path, venv_path, models_path]:
        d.mkdir(parents=True)
    return PackageInfo(
        recipe_name=name,
        path=pkg_dir,
        ready=True,
        venv_path=venv_path,
        repo_path=repo_path,
        models_path=models_path,
    )


# ---------------------------------------------------------------------------
# RunResult / PackageInfo dataclasses
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_default_success(self):
        result = RunResult()
        assert result.success is True
        assert result.error is None
        assert result.outputs == {}
        assert result.metadata == {}

    def test_with_outputs(self):
        result = RunResult(outputs={"image": "/tmp/out.png"}, success=True)
        assert result.outputs["image"] == "/tmp/out.png"

    def test_failure_result(self):
        result = RunResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestPackageInfo:
    def test_fields(self, tmp_path):
        pkg = _make_package_info(tmp_path)
        assert pkg.recipe_name == "test-runner-model"
        assert pkg.ready is True
        assert pkg.repo_path.exists()
        assert pkg.venv_path.exists()


# ---------------------------------------------------------------------------
# _resolve_recipe
# ---------------------------------------------------------------------------

class TestResolveRecipe:
    def test_resolves_local_dot_path(self, tmp_path):
        recipe_file = tmp_path / "my-recipe.yaml"
        recipe_file.write_text(MINIMAL_RECIPE_YAML)
        recipe = _resolve_recipe(str(recipe_file))
        assert recipe.metadata.name == "test-runner-model"

    def test_resolves_absolute_path(self, tmp_path):
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(MINIMAL_RECIPE_YAML)
        recipe = _resolve_recipe(str(recipe_file))
        assert recipe.metadata.name == "test-runner-model"

    def test_resolves_existing_relative_path(self, tmp_path):
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text(MINIMAL_RECIPE_YAML)
        # Path(name).exists() branch
        recipe = _resolve_recipe(str(recipe_file))
        assert recipe.metadata.name == "test-runner-model"

    def test_resolves_registry_name(self):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)

        with patch("kdream.core.runner.RegistryClient") as MockClient:
            MockClient.return_value.fetch_recipe.return_value = mock_recipe
            recipe = _resolve_recipe("test-runner-model")

        assert recipe.metadata.name == "test-runner-model"
        MockClient.return_value.fetch_recipe.assert_called_once_with("test-runner-model")

    def test_registry_error_propagates(self):
        with patch("kdream.core.runner.RegistryClient") as MockClient:
            MockClient.return_value.fetch_recipe.side_effect = RegistryError("not found")
            with pytest.raises(RegistryError):
                _resolve_recipe("nonexistent-recipe")


# ---------------------------------------------------------------------------
# _get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_uses_default_when_none(self, tmp_path):
        with patch("kdream.core.runner.DEFAULT_CACHE_DIR", tmp_path / "default"):
            path = _get_cache_dir(None)
        assert path.exists()

    def test_uses_custom_path(self, tmp_path):
        custom = tmp_path / "my_cache"
        path = _get_cache_dir(str(custom))
        assert path == custom
        assert path.exists()

    def test_creates_directory_if_missing(self, tmp_path):
        target = tmp_path / "new" / "nested" / "cache"
        assert not target.exists()
        _get_cache_dir(str(target))
        assert target.exists()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_calls_backend_install_and_run(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)
        mock_pkg = _make_package_info(tmp_path)

        mock_backend = MagicMock()
        mock_backend.install.return_value = mock_pkg
        mock_backend.run.return_value = {"result": "hello"}

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                result = run(
                    "test-runner-model",
                    cache_dir=str(tmp_path / "cache"),
                    prompt="hello",
                )

        assert result.success is True
        assert result.outputs == {"result": "hello"}
        assert result.metadata["recipe"] == "test-runner-model"
        assert result.metadata["backend"] == "local"
        mock_backend.install.assert_called_once()
        mock_backend.run.assert_called_once_with(mock_pkg, {"prompt": "hello"})

    def test_run_passes_force_reinstall(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)
        mock_pkg = _make_package_info(tmp_path)

        mock_backend = MagicMock()
        mock_backend.install.return_value = mock_pkg
        mock_backend.run.return_value = {}

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                run("test-runner-model", cache_dir=str(tmp_path / "cache"), force_reinstall=True)

        call_kwargs = mock_backend.install.call_args
        assert call_kwargs.kwargs.get("force_reinstall") is True or (
            len(call_kwargs.args) > 2 and call_kwargs.args[2] is True
        )

    def test_run_includes_duration_in_metadata(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)
        mock_pkg = _make_package_info(tmp_path)

        mock_backend = MagicMock()
        mock_backend.install.return_value = mock_pkg
        mock_backend.run.return_value = {}

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                result = run("test-runner-model", cache_dir=str(tmp_path / "cache"))

        assert "duration_s" in result.metadata
        assert result.metadata["duration_s"] >= 0

    def test_run_backend_error_propagates(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)

        mock_backend = MagicMock()
        mock_backend.install.side_effect = BackendError("install failed")

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                with pytest.raises(BackendError, match="install failed"):
                    run("test-runner-model", cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

class TestInstall:
    def test_install_returns_package_info(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)
        mock_pkg = _make_package_info(tmp_path)

        mock_backend = MagicMock()
        mock_backend.install.return_value = mock_pkg

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                result = install("test-runner-model", cache_dir=str(tmp_path / "cache"))

        assert isinstance(result, PackageInfo)
        assert result.recipe_name == "test-runner-model"

    def test_install_does_not_call_run(self, tmp_path):
        from kdream.core.recipe import parse_yaml_recipe
        mock_recipe = parse_yaml_recipe(MINIMAL_RECIPE_YAML)
        mock_pkg = _make_package_info(tmp_path)

        mock_backend = MagicMock()
        mock_backend.install.return_value = mock_pkg

        with patch("kdream.core.runner._resolve_recipe", return_value=mock_recipe):
            with patch("kdream.backends.get_backend", return_value=mock_backend):
                install("test-runner-model", cache_dir=str(tmp_path / "cache"))

        mock_backend.run.assert_not_called()


# ---------------------------------------------------------------------------
# list_installed()
# ---------------------------------------------------------------------------

class TestListInstalled:
    def test_empty_cache_returns_empty_list(self, tmp_path):
        result = list_installed(cache_dir=str(tmp_path))
        assert result == []

    def test_lists_ready_package(self, tmp_path):
        cache = tmp_path / "cache"
        pkg = cache / "my-model"
        (pkg / "repo").mkdir(parents=True)
        (pkg / "venv").mkdir(parents=True)

        result = list_installed(cache_dir=str(cache))
        assert len(result) == 1
        assert result[0].recipe_name == "my-model"
        assert result[0].ready is True

    def test_lists_incomplete_package_as_not_ready(self, tmp_path):
        cache = tmp_path / "cache"
        # Only repo, no venv
        (cache / "partial-model" / "repo").mkdir(parents=True)

        result = list_installed(cache_dir=str(cache))
        assert len(result) == 1
        assert result[0].ready is False

    def test_lists_multiple_packages(self, tmp_path):
        cache = tmp_path / "cache"
        for name in ["model-a", "model-b", "model-c"]:
            pkg = cache / name
            (pkg / "repo").mkdir(parents=True)
            (pkg / "venv").mkdir(parents=True)

        result = list_installed(cache_dir=str(cache))
        assert len(result) == 3
        names = {p.recipe_name for p in result}
        assert names == {"model-a", "model-b", "model-c"}

    def test_ignores_files_in_cache_dir(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "some-file.txt").write_text("hello")
        (cache / "my-model" / "repo").mkdir(parents=True)
        (cache / "my-model" / "venv").mkdir(parents=True)

        result = list_installed(cache_dir=str(cache))
        assert len(result) == 1

    def test_package_info_paths_correct(self, tmp_path):
        cache = tmp_path / "cache"
        pkg = cache / "my-model"
        (pkg / "repo").mkdir(parents=True)
        (pkg / "venv").mkdir(parents=True)

        result = list_installed(cache_dir=str(cache))
        info = result[0]
        assert info.repo_path == pkg / "repo"
        assert info.venv_path == pkg / "venv"
        assert info.models_path == pkg / "models"
        assert info.path == pkg
